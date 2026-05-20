//===--------------------- SIOptimizeVGPRLiveRange.cpp  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass tries to remove unnecessary VGPR live ranges in divergent if-else
/// structures and waterfall loops.
///
/// When we do structurization, we usually transform an if-else into two
/// successive if-then (with a flow block to do predicate inversion). Consider a
/// simple case after structurization: A divergent value %a was defined before
/// if-else and used in both THEN (use in THEN is optional) and ELSE part:
///    bb.if:
///      %a = ...
///      ...
///    bb.then:
///      ... = op %a
///      ... // %a can be dead here
///    bb.flow:
///      ...
///    bb.else:
///      ... = %a
///      ...
///    bb.endif
///
///  As register allocator has no idea of the thread-control-flow, it will just
///  assume %a would be alive in the whole range of bb.then because of a later
///  use in bb.else. On AMDGPU architecture, the VGPR is accessed with respect
///  to exec mask. For this if-else case, the lanes active in bb.then will be
///  inactive in bb.else, and vice-versa. So we are safe to say that %a was dead
///  after the last use in bb.then until the end of the block. The reason is
///  the instructions in bb.then will only overwrite lanes that will never be
///  accessed in bb.else.
///
///  This pass aims to tell register allocator that %a is in-fact dead,
///  through inserting a phi-node in bb.flow saying that %a is undef when coming
///  from bb.then, and then replace the uses in the bb.else with the result of
///  newly inserted phi.
///
///  Two key conditions must be met to ensure correctness:
///  1.) The def-point should be in the same loop-level as if-else-endif to make
///      sure the second loop iteration still get correct data.
///  2.) There should be no further uses after the IF-ELSE region.
///
///
/// Waterfall loops get inserted around instructions that use divergent values
/// but can only be executed with a uniform value. For example an indirect call
/// to a divergent address:
///    bb.start:
///      %a = ...
///      %fun = ...
///      ...
///    bb.loop:
///      call %fun (%a)
///      ... // %a can be dead here
///      loop %bb.loop
///
///  The loop block is executed multiple times, but it is run exactly once for
///  each active lane. Similar to the if-else case, the register allocator
///  assumes that %a is live throughout the loop as it is used again in the next
///  iteration. If %a is a VGPR that is unused after the loop, it does not need
///  to be live after its last use in the loop block. By inserting a phi-node at
///  the start of bb.loop that is undef when coming from bb.loop, the register
///  allocation knows that the value of %a does not need to be preserved through
///  iterations of the loop.
///
/// In addition to the live-range shortening described above, this pass can
/// also optimize placement of v_cmpx instructions inside waterfall loop
/// headers. By default, v_cmpx are deferred to the end of the header to
/// hide latency of v_readfirstlane. To reduce register pressure,
/// this pass can move v_cmpx instruction earlier and
/// place it immediately before the last definition of its source operands.
/// This transformation may shorten live ranges of operands of v_cmpx.
/// Controlled by -amdgpu-waterfall-cmpx-placement-optimization.
/// TODO: Ideally, this transformation would happen in the instruction scheduler,
/// which already has register pressure estimate, but it would require
/// the scheduler to operate across basic blocks and reason about EXEC mask.
//===----------------------------------------------------------------------===//

#include "SIOptimizeVGPRLiveRange.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-opt-vgpr-liverange"

enum class WaterfallCmpxPlacementMode {
  Never,
  Always,
  EstimateRegisterPressure,
};

static cl::opt<WaterfallCmpxPlacementMode> WaterfallCmpxPlacementOptimization(
    "amdgpu-waterfall-cmpx-placement-optimization",
    cl::desc("Control v_cmpx placement in waterfall loop headers"),
    cl::init(WaterfallCmpxPlacementMode::Never), cl::Hidden,
    cl::values(
        clEnumValN(WaterfallCmpxPlacementMode::Never, "never",
                   "Keep cmpx deferred at end of header (default)"),
        clEnumValN(WaterfallCmpxPlacementMode::Always, "always",
                   "Always move cmpx earlier"),
        clEnumValN(WaterfallCmpxPlacementMode::EstimateRegisterPressure,
                   "estimate-register-pressure",
                   "Move cmpx earlier only when estimated VGPR pressure in "
                   "the header exceeds the target")));

namespace {

class SIOptimizeVGPRLiveRange {
private:
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  LiveVariables *LV = nullptr;
  MachineDominatorTree *MDT = nullptr;
  MachineLoopInfo *Loops = nullptr;
  MachineRegisterInfo *MRI = nullptr;

public:
  SIOptimizeVGPRLiveRange(LiveVariables *LV, MachineDominatorTree *MDT,
                          MachineLoopInfo *Loops)
      : LV(LV), MDT(MDT), Loops(Loops) {}
  bool run(MachineFunction &MF);

  MachineBasicBlock *getElseTarget(MachineBasicBlock *MBB) const;

  void collectElseRegionBlocks(MachineBasicBlock *Flow,
                               MachineBasicBlock *Endif,
                               SmallSetVector<MachineBasicBlock *, 16> &) const;

  void
  collectCandidateRegisters(MachineBasicBlock *If, MachineBasicBlock *Flow,
                            MachineBasicBlock *Endif,
                            SmallSetVector<MachineBasicBlock *, 16> &ElseBlocks,
                            SmallVectorImpl<Register> &CandidateRegs) const;

  void collectWaterfallCandidateRegisters(
      MachineBasicBlock *LoopHeader, MachineBasicBlock *LoopEnd,
      SmallSetVector<Register, 16> &CandidateRegs,
      SmallSetVector<MachineBasicBlock *, 2> &Blocks,
      SmallVectorImpl<MachineInstr *> &Instructions) const;

  void findNonPHIUsesInBlock(Register Reg, MachineBasicBlock *MBB,
                             SmallVectorImpl<MachineInstr *> &Uses) const;

  void updateLiveRangeInThenRegion(Register Reg, MachineBasicBlock *If,
                                   MachineBasicBlock *Flow) const;

  void updateLiveRangeInElseRegion(
      Register Reg, Register NewReg, MachineBasicBlock *Flow,
      MachineBasicBlock *Endif,
      SmallSetVector<MachineBasicBlock *, 16> &ElseBlocks) const;

  void
  optimizeLiveRange(Register Reg, MachineBasicBlock *If,
                    MachineBasicBlock *Flow, MachineBasicBlock *Endif,
                    SmallSetVector<MachineBasicBlock *, 16> &ElseBlocks) const;

  void optimizeWaterfallLiveRange(
      Register Reg, MachineBasicBlock *LoopHeader,
      SmallSetVector<MachineBasicBlock *, 2> &LoopBlocks,
      SmallVectorImpl<MachineInstr *> &Instructions) const;

  using HeaderWithLiveThrough =
      std::pair<MachineBasicBlock *, SmallVector<Register, 16>>;
  void collectWaterfallCandidateHeaders(
      MachineFunction &MF,
      SmallVectorImpl<HeaderWithLiveThrough> &HeadersToReorder) const;
  bool optimizeWaterfallCmpxPlacement(MachineBasicBlock *LoopHeader,
                                      ArrayRef<Register> LiveThrough);
  void updateLiveVariablesAfterBlockSplit(
      MachineBasicBlock *CurBB, MachineBasicBlock *NewBB,
      SmallVectorImpl<Register> &LiveThrough) const;
};

class SIOptimizeVGPRLiveRangeLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIOptimizeVGPRLiveRangeLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Optimize VGPR LiveRange";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveVariablesWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<LiveVariablesWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setNoPHIs();
  }
};

} // end anonymous namespace

// Check whether the MBB is a else flow block and get the branching target which
// is the Endif block
MachineBasicBlock *
SIOptimizeVGPRLiveRange::getElseTarget(MachineBasicBlock *MBB) const {
  for (auto &BR : MBB->terminators()) {
    if (BR.getOpcode() == AMDGPU::SI_ELSE)
      return BR.getOperand(2).getMBB();
  }
  return nullptr;
}

void SIOptimizeVGPRLiveRange::collectElseRegionBlocks(
    MachineBasicBlock *Flow, MachineBasicBlock *Endif,
    SmallSetVector<MachineBasicBlock *, 16> &Blocks) const {
  assert(Flow != Endif);

  MachineBasicBlock *MBB = Endif;
  unsigned Cur = 0;
  while (MBB) {
    for (auto *Pred : MBB->predecessors()) {
      if (Pred != Flow)
        Blocks.insert(Pred);
    }

    if (Cur < Blocks.size())
      MBB = Blocks[Cur++];
    else
      MBB = nullptr;
  }

  LLVM_DEBUG({
    dbgs() << "Found Else blocks: ";
    for (auto *MBB : Blocks)
      dbgs() << printMBBReference(*MBB) << ' ';
    dbgs() << '\n';
  });
}

/// Find the instructions(excluding phi) in \p MBB that uses the \p Reg.
void SIOptimizeVGPRLiveRange::findNonPHIUsesInBlock(
    Register Reg, MachineBasicBlock *MBB,
    SmallVectorImpl<MachineInstr *> &Uses) const {
  for (auto &UseMI : MRI->use_nodbg_instructions(Reg)) {
    if (UseMI.getParent() == MBB && !UseMI.isPHI())
      Uses.push_back(&UseMI);
  }
}

/// Collect the killed registers in the ELSE region which are not alive through
/// the whole THEN region.
void SIOptimizeVGPRLiveRange::collectCandidateRegisters(
    MachineBasicBlock *If, MachineBasicBlock *Flow, MachineBasicBlock *Endif,
    SmallSetVector<MachineBasicBlock *, 16> &ElseBlocks,
    SmallVectorImpl<Register> &CandidateRegs) const {

  SmallSet<Register, 8> KillsInElse;

  for (auto *Else : ElseBlocks) {
    for (auto &MI : Else->instrs()) {
      if (MI.isDebugInstr())
        continue;

      for (auto &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || MO.isDef())
          continue;

        Register MOReg = MO.getReg();
        // We can only optimize AGPR/VGPR virtual register
        if (MOReg.isPhysical() || !TRI->isVectorRegister(*MRI, MOReg))
          continue;

        if (MO.readsReg()) {
          LiveVariables::VarInfo &VI = LV->getVarInfo(MOReg);
          const MachineBasicBlock *DefMBB = MRI->getVRegDef(MOReg)->getParent();
          // Make sure two conditions are met:
          // a.) the value is defined before/in the IF block
          // b.) should be defined in the same loop-level.
          if ((VI.AliveBlocks.test(If->getNumber()) || DefMBB == If) &&
              Loops->getLoopFor(DefMBB) == Loops->getLoopFor(If)) {
            // Check if the register is live into the endif block. If not,
            // consider it killed in the else region.
            LiveVariables::VarInfo &VI = LV->getVarInfo(MOReg);
            if (!VI.isLiveIn(*Endif, MOReg, *MRI)) {
              KillsInElse.insert(MOReg);
            } else {
              LLVM_DEBUG(dbgs() << "Excluding " << printReg(MOReg, TRI)
                                << " as Live in Endif\n");
            }
          }
        }
      }
    }
  }

  // Check the phis in the Endif, looking for value coming from the ELSE
  // region. Make sure the phi-use is the last use.
  for (auto &MI : Endif->phis()) {
    for (unsigned Idx = 1; Idx < MI.getNumOperands(); Idx += 2) {
      auto &MO = MI.getOperand(Idx);
      auto *Pred = MI.getOperand(Idx + 1).getMBB();
      if (Pred == Flow)
        continue;
      assert(ElseBlocks.contains(Pred) && "Should be from Else region\n");

      if (!MO.isReg() || !MO.getReg() || MO.isUndef())
        continue;

      Register Reg = MO.getReg();
      if (Reg.isPhysical() || !TRI->isVectorRegister(*MRI, Reg))
        continue;

      LiveVariables::VarInfo &VI = LV->getVarInfo(Reg);

      if (VI.isLiveIn(*Endif, Reg, *MRI)) {
        LLVM_DEBUG(dbgs() << "Excluding " << printReg(Reg, TRI)
                          << " as Live in Endif\n");
        continue;
      }
      // Make sure two conditions are met:
      // a.) the value is defined before/in the IF block
      // b.) should be defined in the same loop-level.
      const MachineBasicBlock *DefMBB = MRI->getVRegDef(Reg)->getParent();
      if ((VI.AliveBlocks.test(If->getNumber()) || DefMBB == If) &&
          Loops->getLoopFor(DefMBB) == Loops->getLoopFor(If))
        KillsInElse.insert(Reg);
    }
  }

  auto IsLiveThroughThen = [&](Register Reg) {
    for (auto I = MRI->use_nodbg_begin(Reg), E = MRI->use_nodbg_end(); I != E;
         ++I) {
      if (!I->readsReg())
        continue;
      auto *UseMI = I->getParent();
      auto *UseMBB = UseMI->getParent();
      if (UseMBB == Flow || UseMBB == Endif) {
        if (!UseMI->isPHI())
          return true;

        auto *IncomingMBB = UseMI->getOperand(I.getOperandNo() + 1).getMBB();
        // The register is live through the path If->Flow or Flow->Endif.
        // we should not optimize for such cases.
        if ((UseMBB == Flow && IncomingMBB != If) ||
            (UseMBB == Endif && IncomingMBB == Flow))
          return true;
      }
    }
    return false;
  };

  for (auto Reg : KillsInElse) {
    if (!IsLiveThroughThen(Reg))
      CandidateRegs.push_back(Reg);
  }
}

/// Collect the registers used in the waterfall loop block that are defined
/// before.
void SIOptimizeVGPRLiveRange::collectWaterfallCandidateRegisters(
    MachineBasicBlock *LoopHeader, MachineBasicBlock *LoopEnd,
    SmallSetVector<Register, 16> &CandidateRegs,
    SmallSetVector<MachineBasicBlock *, 2> &Blocks,
    SmallVectorImpl<MachineInstr *> &Instructions) const {

  // Collect loop instructions, potentially spanning multiple blocks
  auto *MBB = LoopHeader;
  for (;;) {
    Blocks.insert(MBB);
    for (auto &MI : *MBB) {
      if (MI.isDebugInstr())
        continue;
      Instructions.push_back(&MI);
    }
    if (MBB == LoopEnd)
      break;

    if ((MBB != LoopHeader && MBB->pred_size() != 1) ||
        (MBB == LoopHeader && MBB->pred_size() != 2) || MBB->succ_size() != 1) {
      LLVM_DEBUG(dbgs() << "Unexpected edges in CFG, ignoring loop\n");
      return;
    }

    MBB = *MBB->succ_begin();
  }

  for (auto *I : Instructions) {
    auto &MI = *I;

    for (auto &MO : MI.all_uses()) {
      if (!MO.getReg())
        continue;

      Register MOReg = MO.getReg();
      // We can only optimize AGPR/VGPR virtual register
      if (MOReg.isPhysical() || !TRI->isVectorRegister(*MRI, MOReg))
        continue;

      if (MO.readsReg()) {
        MachineBasicBlock *DefMBB = MRI->getVRegDef(MOReg)->getParent();
        // Make sure the value is defined before the LOOP block
        if (!Blocks.contains(DefMBB) && !CandidateRegs.contains(MOReg)) {
          // If the variable is used after the loop, the register coalescer will
          // merge the newly created register and remove the phi node again.
          // Just do nothing in that case.
          LiveVariables::VarInfo &OldVarInfo = LV->getVarInfo(MOReg);
          bool IsUsed = false;
          for (auto *Succ : LoopEnd->successors()) {
            if (!Blocks.contains(Succ) &&
                OldVarInfo.isLiveIn(*Succ, MOReg, *MRI)) {
              IsUsed = true;
              break;
            }
          }
          if (!IsUsed) {
            LLVM_DEBUG(dbgs() << "Found candidate reg: "
                              << printReg(MOReg, TRI, 0, MRI) << '\n');
            CandidateRegs.insert(MOReg);
          } else {
            LLVM_DEBUG(dbgs() << "Reg is used after loop, ignoring: "
                              << printReg(MOReg, TRI, 0, MRI) << '\n');
          }
        }
      }
    }
  }
}

// Re-calculate the liveness of \p Reg in the THEN-region
void SIOptimizeVGPRLiveRange::updateLiveRangeInThenRegion(
    Register Reg, MachineBasicBlock *If, MachineBasicBlock *Flow) const {
  SetVector<MachineBasicBlock *> Blocks;
  SmallVector<MachineBasicBlock *> WorkList({If});

  // Collect all successors until we see the flow block, where we should
  // reconverge.
  while (!WorkList.empty()) {
    auto *MBB = WorkList.pop_back_val();
    for (auto *Succ : MBB->successors()) {
      if (Succ != Flow && Blocks.insert(Succ))
        WorkList.push_back(Succ);
    }
  }

  LiveVariables::VarInfo &OldVarInfo = LV->getVarInfo(Reg);
  for (MachineBasicBlock *MBB : Blocks) {
    // Clear Live bit, as we will recalculate afterwards
    LLVM_DEBUG(dbgs() << "Clear AliveBlock " << printMBBReference(*MBB)
                      << '\n');
    OldVarInfo.AliveBlocks.reset(MBB->getNumber());
  }

  SmallPtrSet<MachineBasicBlock *, 4> PHIIncoming;

  // Get the blocks the Reg should be alive through
  for (auto I = MRI->use_nodbg_begin(Reg), E = MRI->use_nodbg_end(); I != E;
       ++I) {
    auto *UseMI = I->getParent();
    if (UseMI->isPHI() && I->readsReg()) {
      if (Blocks.contains(UseMI->getParent()))
        PHIIncoming.insert(UseMI->getOperand(I.getOperandNo() + 1).getMBB());
    }
  }

  for (MachineBasicBlock *MBB : Blocks) {
    SmallVector<MachineInstr *> Uses;
    // PHI instructions has been processed before.
    findNonPHIUsesInBlock(Reg, MBB, Uses);

    if (Uses.size() == 1) {
      LLVM_DEBUG(dbgs() << "Found one Non-PHI use in "
                        << printMBBReference(*MBB) << '\n');
      LV->HandleVirtRegUse(Reg, MBB, *(*Uses.begin()));
    } else if (Uses.size() > 1) {
      // Process the instructions in-order
      LLVM_DEBUG(dbgs() << "Found " << Uses.size() << " Non-PHI uses in "
                        << printMBBReference(*MBB) << '\n');
      for (MachineInstr &MI : *MBB) {
        if (llvm::is_contained(Uses, &MI))
          LV->HandleVirtRegUse(Reg, MBB, MI);
      }
    }

    // Mark Reg alive through the block if this is a PHI incoming block
    if (PHIIncoming.contains(MBB))
      LV->MarkVirtRegAliveInBlock(OldVarInfo, MRI->getVRegDef(Reg)->getParent(),
                                  MBB);
  }

  // Set the isKilled flag if we get new Kills in the THEN region.
  for (auto *MI : OldVarInfo.Kills) {
    if (Blocks.contains(MI->getParent()))
      MI->addRegisterKilled(Reg, TRI);
  }
}

void SIOptimizeVGPRLiveRange::updateLiveRangeInElseRegion(
    Register Reg, Register NewReg, MachineBasicBlock *Flow,
    MachineBasicBlock *Endif,
    SmallSetVector<MachineBasicBlock *, 16> &ElseBlocks) const {
  LiveVariables::VarInfo &NewVarInfo = LV->getVarInfo(NewReg);
  LiveVariables::VarInfo &OldVarInfo = LV->getVarInfo(Reg);

  // Transfer aliveBlocks from Reg to NewReg
  for (auto *MBB : ElseBlocks) {
    unsigned BBNum = MBB->getNumber();
    if (OldVarInfo.AliveBlocks.test(BBNum)) {
      NewVarInfo.AliveBlocks.set(BBNum);
      LLVM_DEBUG(dbgs() << "Removing AliveBlock " << printMBBReference(*MBB)
                        << '\n');
      OldVarInfo.AliveBlocks.reset(BBNum);
    }
  }

  // Transfer the possible Kills in ElseBlocks from Reg to NewReg
  auto I = OldVarInfo.Kills.begin();
  while (I != OldVarInfo.Kills.end()) {
    if (ElseBlocks.contains((*I)->getParent())) {
      NewVarInfo.Kills.push_back(*I);
      I = OldVarInfo.Kills.erase(I);
    } else {
      ++I;
    }
  }
}

void SIOptimizeVGPRLiveRange::optimizeLiveRange(
    Register Reg, MachineBasicBlock *If, MachineBasicBlock *Flow,
    MachineBasicBlock *Endif,
    SmallSetVector<MachineBasicBlock *, 16> &ElseBlocks) const {
  // Insert a new PHI, marking the value from the THEN region being
  // undef.
  LLVM_DEBUG(dbgs() << "Optimizing " << printReg(Reg, TRI) << '\n');
  const auto *RC = MRI->getRegClass(Reg);
  Register NewReg = MRI->createVirtualRegister(RC);
  Register UndefReg = MRI->createVirtualRegister(RC);
  MachineInstrBuilder PHI = BuildMI(*Flow, Flow->getFirstNonPHI(), DebugLoc(),
                                    TII->get(TargetOpcode::PHI), NewReg);
  for (auto *Pred : Flow->predecessors()) {
    if (Pred == If)
      PHI.addReg(Reg).addMBB(Pred);
    else
      PHI.addReg(UndefReg, RegState::Undef).addMBB(Pred);
  }

  // Replace all uses in the ELSE region or the PHIs in ENDIF block
  // Use early increment range because setReg() will update the linked list.
  for (auto &O : make_early_inc_range(MRI->use_operands(Reg))) {
    auto *UseMI = O.getParent();
    auto *UseBlock = UseMI->getParent();
    // Replace uses in Endif block
    if (UseBlock == Endif) {
      if (UseMI->isPHI())
        O.setReg(NewReg);
      else if (UseMI->isDebugInstr())
        continue;
      else {
        // DetectDeadLanes may mark register uses as undef without removing
        // them, in which case a non-phi instruction using the original register
        // may exist in the Endif block even though the register is not live
        // into it.
        assert(!O.readsReg());
      }
      continue;
    }

    // Replace uses in Else region
    if (ElseBlocks.contains(UseBlock))
      O.setReg(NewReg);
  }

  // The optimized Reg is not alive through Flow blocks anymore.
  LiveVariables::VarInfo &OldVarInfo = LV->getVarInfo(Reg);
  OldVarInfo.AliveBlocks.reset(Flow->getNumber());

  updateLiveRangeInElseRegion(Reg, NewReg, Flow, Endif, ElseBlocks);
  updateLiveRangeInThenRegion(Reg, If, Flow);
}

static inline bool isCmpxTermOpcode(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::V_CMPX_EQ_U32_nosdst_e32_term:
  case AMDGPU::V_CMPX_EQ_U32_nosdst_e64_term:
  case AMDGPU::V_CMPX_EQ_U64_nosdst_e32_term:
  case AMDGPU::V_CMPX_EQ_U64_nosdst_e64_term:
    return true;
  default:
    return false;
  }
}

static inline bool hasKilledVirtRegOperand(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.uses()) {
    if (MO.isReg() && MO.getReg().isVirtual() && MO.isKill())
      return true;
  }
  return false;
}

static inline bool isOperandExpected(const MachineOperand &MO) {
  if (MO.isImm() || MO.isFPImm() || MO.isCImm())
    return true;
  if (MO.isReg() && MO.getReg().isVirtual())
    return true;
  return false;
}

static inline bool isNonTerminatorExpected(MachineInstr &NT) {
  switch (NT.getOpcode()) {
    case TargetOpcode::PHI:
    case TargetOpcode::REG_SEQUENCE:
    case AMDGPU::V_READFIRSTLANE_B32:
      return true;
    default:
      return false;
    }
}

static inline bool isWaterfallCandidateHeader(MachineBasicBlock *Header) {
  unsigned NumCmpx = 0;
  bool CmpxHasKilledOperand = false;
  for (MachineInstr &MI : *Header) {
    if (!MI.isTerminator()) {
      if (!isNonTerminatorExpected(MI))
        return false;
    }
    else {
      if (!isCmpxTermOpcode(MI.getOpcode()))
        return false;

      for (const MachineOperand &MO : MI.explicit_uses()) {
        if (!isOperandExpected(MO))
          return false;
      }

      ++NumCmpx;
      if (!CmpxHasKilledOperand && hasKilledVirtRegOperand(MI))
        CmpxHasKilledOperand = true;
    }
  }
  return (CmpxHasKilledOperand && NumCmpx >= 2);
}

// Collect waterfall loop headers that may benefit from reordering and for each
// selected header return its live-through vreg list.
void SIOptimizeVGPRLiveRange::collectWaterfallCandidateHeaders(
    MachineFunction &MF,
    SmallVectorImpl<HeaderWithLiveThrough> &HeadersToReorder) const {
  const bool EstimatePressure =
      WaterfallCmpxPlacementOptimization ==
      WaterfallCmpxPlacementMode::EstimateRegisterPressure;
  assert((EstimatePressure || WaterfallCmpxPlacementOptimization ==
                                  WaterfallCmpxPlacementMode::Always) &&
         "caller must check that the optimization is enabled");

  // For safety, a candidate header must have the exact shape
  // produced by previuos passes that insert waterfall loops.
  //   - non-terminators are only PHI, V_READFIRSTLANE_B32, and REG_SEQUENCE
  //   - terminators are all cmpx_term
  // Moving cmpx earlier is possible and would shortens the live range of at least
  // one virtual register:
  //   - at least two cmpx terminators
  //   - at leastone with a killed virtual-register source operand
  SmallVector<MachineBasicBlock *, 4> CandidateHeaders;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.terminators()) {
      if (MI.getOpcode() != AMDGPU::SI_WATERFALL_LOOP)
        continue;
      MachineBasicBlock *Header = MI.getOperand(0).getMBB();
      if (isWaterfallCandidateHeader(Header))
        CandidateHeaders.push_back(Header);
    }
  }
  if (CandidateHeaders.empty())
    return;

  SmallVector<SmallVector<Register, 16>, 4> LiveThrough(CandidateHeaders.size());
  SmallVector<GCNRegPressure, 4> RP;
  if (EstimatePressure)
    RP.resize(CandidateHeaders.size());

  // A single pass over all virtual registers
  // (a) estimates register pressure of the candidate headers
  // (b) computes the live-through contribution of the candidate headers so that it can be
  // updated incrementally when cmpx reordering forces block splitting
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI->def_empty(Reg))
      continue;
    LiveVariables::VarInfo &VI = LV->getVarInfo(Reg);
    for (auto [Idx, Header] : enumerate(CandidateHeaders)) {
      if (!VI.AliveBlocks.test(Header->getNumber()))
        continue;
      LiveThrough[Idx].push_back(Reg);
      if (EstimatePressure)
        RP[Idx].inc(Reg, LaneBitmask::getNone(), MRI->getMaxLaneMaskForVReg(Reg), *MRI);
    }
  }

  // For each candidate header, check register pressure and if selected for
  // reordering, append the header to the result along with its live-through
  // vreg list
  for (auto [Idx, Header] : enumerate(CandidateHeaders)) {
    if (EstimatePressure) {
      // Add per-block contributions from defs inside the header
      for (MachineInstr &MI : *Header) {
        for (const MachineOperand &MO : MI.defs()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          RP[Idx].inc(MO.getReg(), LaneBitmask::getNone(),
                      MRI->getMaxLaneMaskForVReg(MO.getReg()), *MRI);
        }
      }
      GCNRPTarget Target(MF, RP[Idx]);
      bool High = !Target.satisfied(RP[Idx]);
      LLVM_DEBUG(dbgs() << "Register pressure in " << printMBBReference(*Header)
                        << ": SGPRs=" << RP[Idx].getSGPRNum()
                        << " VGPRs=" << RP[Idx].getArchVGPRNum()
                        << (High ? " (high)\n" : " (ok)\n"));
      if (!High)
        continue;
    }
    HeadersToReorder.emplace_back(Header, std::move(LiveThrough[Idx]));
  }
}

bool SIOptimizeVGPRLiveRange::optimizeWaterfallCmpxPlacement(
    MachineBasicBlock *LoopHeader, ArrayRef<Register> LiveThroughIn) {
  MachineFunction &MF = *LoopHeader->getParent();
  SmallVector<MachineInstr *, 4> Candidates;
  for (MachineInstr &MI : LoopHeader->terminators()) {
    assert(isCmpxTermOpcode(MI.getOpcode()) &&
           "waterfall header should only have cmpx_term terminators");
    Candidates.push_back(&MI);
  }

  LLVM_DEBUG(dbgs() << "Moving " << Candidates.size()
                    << " cmpx_term instructions in "
                    << printMBBReference(*LoopHeader) << '\n');

  // Mutable copy of the live-through set; we maintain it incrementally
  // across splits, dropping any reg defined or killed inside the lower half.
  SmallVector<Register, 16> LiveThrough(LiveThroughIn.begin(),
                                        LiveThroughIn.end());

  // For each candidate, find the last instruction that defines any of its
  // source operands, and move the cmpx to immediately after it. Block
  // splitting is required because cmpx_term writes EXEC.
  MachineBasicBlock *CurBB = LoopHeader;

  for (MachineInstr *Cmpx : Candidates) {

    // Find the last def of any source operand in the non-terminator region.
    MachineBasicBlock::iterator InsertPt;
    bool Found = false;
    for (auto It = CurBB->getFirstTerminator(); It != CurBB->begin();) {
      --It;
      if (It->isPHI())
        break;
      for (const MachineOperand &MO : Cmpx->uses()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        if (It->definesRegister(MO.getReg(), TRI)) {
          InsertPt = std::next(It);
          Found = true;
          break;
        }
      }
      if (Found)
        break;
    }

    if (!Found)
      continue;

    CurBB->splice(InsertPt, CurBB, Cmpx);
    LLVM_DEBUG(dbgs() << "  Moved: " << Cmpx);

    // Split the block after the cmpx_term.
    MachineBasicBlock *NewBB = MF.CreateMachineBasicBlock();
    MF.insert(std::next(CurBB->getIterator()), NewBB);

    NewBB->splice(NewBB->end(), CurBB, InsertPt, CurBB->end());
    NewBB->transferSuccessorsAndUpdatePHIs(CurBB);
    CurBB->addSuccessor(NewBB);

    if (MDT) {
      MDT->addNewBlock(NewBB, CurBB);
      for (MachineBasicBlock *Succ : NewBB->successors()) {
        if (MDT->dominates(CurBB, Succ))
          MDT->changeImmediateDominator(Succ, NewBB);
      }
    }

    if (Loops) {
      if (MachineLoop *L = Loops->getLoopFor(CurBB))
        L->addBasicBlockToLoop(NewBB, *Loops);
    }

    updateLiveVariablesAfterBlockSplit(CurBB, NewBB, LiveThrough);

    CurBB = NewBB;
  }

  return true;
}

// Update LiveVariables after splitting CurBB into (updated CurBB, NewBB) by
// moving its lower half into NewBB. The split changes liveness in two ways:
//
//   Case A: regs live-through original CurBB. After the split, they are still
//   live-in to the updated CurBB (already in AliveBlocks). If not touched in
//   NewBB, they are also live-through NewBB and need AliveBlocks[NewBB]; if
//   killed in NewBB, they are no longer live-through (LV's Kills MI pointers
//   stay valid across reparenting).
//
//   Case B: regs killed in original CurBB without a local def. The kill MI
//   may have moved into NewBB; the reg is then live-in to updated CurBB and
//   passes through to NewBB, so AliveBlocks[updated CurBB] must be set.
//
// LiveThrough is the per-header live-through set; defs and kills inside NewBB
// are dropped from it so the next split iteration sees the right set.
void SIOptimizeVGPRLiveRange::updateLiveVariablesAfterBlockSplit(
    MachineBasicBlock *CurBB, MachineBasicBlock *NewBB,
    SmallVectorImpl<Register> &LiveThrough) const {
  SmallDenseSet<Register, 8> DefsInNewBB;
  SmallDenseSet<Register, 8> KillsInNewBB;
  for (MachineInstr &MI : *NewBB) {
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;
      if (MO.isDef())
        DefsInNewBB.insert(MO.getReg());
      else if (MO.isKill())
        KillsInNewBB.insert(MO.getReg());
    }
  }
  SmallDenseSet<Register, 8> DefsInCurBB;
  for (MachineInstr &MI : *CurBB) {
    for (const MachineOperand &MO : MI.defs()) {
      if (MO.isReg() && MO.getReg().isVirtual())
        DefsInCurBB.insert(MO.getReg());
    }
  }

  unsigned NewNum = NewBB->getNumber();
  unsigned CurNum = CurBB->getNumber();

  // Case A: drop defs and kills from LiveThrough; propagate the rest.
  auto *NewEnd = std::remove_if(
      LiveThrough.begin(), LiveThrough.end(), [&](Register R) {
        if (DefsInNewBB.contains(R))
          return true;
        if (KillsInNewBB.contains(R))
          return true;
        LV->getVarInfo(R).AliveBlocks.set(NewNum);
        return false;
      });
  LiveThrough.erase(NewEnd, LiveThrough.end());

  // Case B: any reg killed in NewBB whose def is in neither updated CurBB nor
  // NewBB must be live-in to updated CurBB as a pass-through.
  for (Register R : KillsInNewBB) {
    if (DefsInNewBB.contains(R) || DefsInCurBB.contains(R))
      continue;
    LV->getVarInfo(R).AliveBlocks.set(CurNum);
  }
}

void SIOptimizeVGPRLiveRange::optimizeWaterfallLiveRange(
    Register Reg, MachineBasicBlock *LoopHeader,
    SmallSetVector<MachineBasicBlock *, 2> &Blocks,
    SmallVectorImpl<MachineInstr *> &Instructions) const {
  // Insert a new PHI, marking the value from the last loop iteration undef.
  LLVM_DEBUG(dbgs() << "Optimizing " << printReg(Reg, TRI) << '\n');
  const auto *RC = MRI->getRegClass(Reg);
  Register NewReg = MRI->createVirtualRegister(RC);
  Register UndefReg = MRI->createVirtualRegister(RC);

  // Replace all uses in the LOOP region
  // Use early increment range because setReg() will update the linked list.
  for (auto &O : make_early_inc_range(MRI->use_operands(Reg))) {
    auto *UseMI = O.getParent();
    auto *UseBlock = UseMI->getParent();
    // Replace uses in Loop blocks
    if (Blocks.contains(UseBlock))
      O.setReg(NewReg);
  }

  MachineInstrBuilder PHI =
      BuildMI(*LoopHeader, LoopHeader->getFirstNonPHI(), DebugLoc(),
              TII->get(TargetOpcode::PHI), NewReg);
  for (auto *Pred : LoopHeader->predecessors()) {
    if (Blocks.contains(Pred))
      PHI.addReg(UndefReg, RegState::Undef).addMBB(Pred);
    else
      PHI.addReg(Reg).addMBB(Pred);
  }

  LiveVariables::VarInfo &NewVarInfo = LV->getVarInfo(NewReg);
  LiveVariables::VarInfo &OldVarInfo = LV->getVarInfo(Reg);

  // Find last use and mark as kill
  MachineInstr *Kill = nullptr;
  for (auto *MI : reverse(Instructions)) {
    if (MI->readsRegister(NewReg, TRI)) {
      MI->addRegisterKilled(NewReg, TRI);
      NewVarInfo.Kills.push_back(MI);
      Kill = MI;
      break;
    }
  }
  assert(Kill && "Failed to find last usage of register in loop");

  MachineBasicBlock *KillBlock = Kill->getParent();
  bool PostKillBlock = false;
  for (auto *Block : Blocks) {
    auto BBNum = Block->getNumber();

    // collectWaterfallCandidateRegisters only collects registers that are dead
    // after the loop. So we know that the old reg is no longer live throughout
    // the waterfall loop.
    OldVarInfo.AliveBlocks.reset(BBNum);

    // The new register is live up to (and including) the block that kills it.
    PostKillBlock |= (Block == KillBlock);
    if (PostKillBlock) {
      NewVarInfo.AliveBlocks.reset(BBNum);
    } else if (Block != LoopHeader) {
      NewVarInfo.AliveBlocks.set(BBNum);
    }
  }
}

char SIOptimizeVGPRLiveRangeLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(SIOptimizeVGPRLiveRangeLegacy, DEBUG_TYPE,
                      "SI Optimize VGPR LiveRange", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_END(SIOptimizeVGPRLiveRangeLegacy, DEBUG_TYPE,
                    "SI Optimize VGPR LiveRange", false, false)

char &llvm::SIOptimizeVGPRLiveRangeLegacyID = SIOptimizeVGPRLiveRangeLegacy::ID;

FunctionPass *llvm::createSIOptimizeVGPRLiveRangeLegacyPass() {
  return new SIOptimizeVGPRLiveRangeLegacy();
}

bool SIOptimizeVGPRLiveRangeLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  LiveVariables *LV = &getAnalysis<LiveVariablesWrapperPass>().getLV();
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachineLoopInfo *Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return SIOptimizeVGPRLiveRange(LV, MDT, Loops).run(MF);
}

PreservedAnalyses
SIOptimizeVGPRLiveRangePass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  LiveVariables *LV = &MFAM.getResult<LiveVariablesAnalysis>(MF);
  MachineDominatorTree *MDT = &MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachineLoopInfo *Loops = &MFAM.getResult<MachineLoopAnalysis>(MF);

  bool Changed = SIOptimizeVGPRLiveRange(LV, MDT, Loops).run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<LiveVariablesAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<MachineLoopAnalysis>();
  return PA;
}

bool SIOptimizeVGPRLiveRange::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool MadeChange = false;

  // TODO: we need to think about the order of visiting the blocks to get
  // optimal result for nesting if-else cases.
  for (MachineBasicBlock &MBB : MF) {
    for (auto &MI : MBB.terminators()) {
      // Detect the if-else blocks
      if (MI.getOpcode() == AMDGPU::SI_IF) {
        MachineBasicBlock *IfTarget = MI.getOperand(2).getMBB();
        auto *Endif = getElseTarget(IfTarget);
        if (!Endif)
          continue;

        // Skip unexpected control flow.
        if (!MDT->dominates(&MBB, IfTarget) || !MDT->dominates(IfTarget, Endif))
          continue;

        SmallSetVector<MachineBasicBlock *, 16> ElseBlocks;
        SmallVector<Register> CandidateRegs;

        LLVM_DEBUG(dbgs() << "Checking IF-ELSE-ENDIF: "
                          << printMBBReference(MBB) << ' '
                          << printMBBReference(*IfTarget) << ' '
                          << printMBBReference(*Endif) << '\n');

        // Collect all the blocks in the ELSE region
        collectElseRegionBlocks(IfTarget, Endif, ElseBlocks);

        // Collect the registers can be optimized
        collectCandidateRegisters(&MBB, IfTarget, Endif, ElseBlocks,
                                  CandidateRegs);
        MadeChange |= !CandidateRegs.empty();
        // Now we are safe to optimize.
        for (auto Reg : CandidateRegs)
          optimizeLiveRange(Reg, &MBB, IfTarget, Endif, ElseBlocks);
      } else if (MI.getOpcode() == AMDGPU::SI_WATERFALL_LOOP) {
        auto *LoopHeader = MI.getOperand(0).getMBB();
        auto *LoopEnd = &MBB;

        LLVM_DEBUG(dbgs() << "Checking Waterfall loop: "
                          << printMBBReference(*LoopHeader) << '\n');

        SmallSetVector<Register, 16> CandidateRegs;
        SmallVector<MachineInstr *, 16> Instructions;
        SmallSetVector<MachineBasicBlock *, 2> Blocks;

        collectWaterfallCandidateRegisters(LoopHeader, LoopEnd, CandidateRegs,
                                           Blocks, Instructions);
        MadeChange |= !CandidateRegs.empty();
        // Now we are safe to optimize.
        for (auto Reg : CandidateRegs)
          optimizeWaterfallLiveRange(Reg, LoopHeader, Blocks, Instructions);
      }
    }
  }

  // Optimize placement of v_cmpx instructions inside waterfall loop
  // headers. By default, v_cmpx are deferred to the end of the header to
  // hide latency of v_readfirstlane instructions they depend on.
  // To reduce register pressure, this pass can move v_cmpx instruction earlier and
  // place it immediately before the last definition of its source operands.
  if (WaterfallCmpxPlacementOptimization != WaterfallCmpxPlacementMode::Never) {
    // Incrementally updates LiveVariables
    SmallVector<HeaderWithLiveThrough, 4> HeadersToReorder;
    collectWaterfallCandidateHeaders(MF, HeadersToReorder);
    for (auto &[Header, LiveThrough] : HeadersToReorder)
      MadeChange |= optimizeWaterfallCmpxPlacement(Header, LiveThrough);
  }

  return MadeChange;
}
