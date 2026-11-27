# Wave64 programs on hosts of 32-lane waves

The console runs compute programs in waves of 64 lanes. NVIDIA and Intel
run waves of 32 at most (`WaveLaneCountMax` 32) and refuse `WaveSize(64)`.
This note describes how a console wave runs on such a host, what each
cross-lane operation needs there, and what is measured on the title. It
covers compute programs; vertex, pixel and mesh programs do not declare
the console's wave size.

## Halves

A wave64 program whose lanes must be the console's (`Lower::console_lanes`)
is translated for waves of 32 (`emit::Options::host_wave32`,
`emit::lower::halves`). The thread relabelling gives each host wave a
32-aligned run of thread ids, so host wave `2w + h` holds lanes
`32h..32h + 31` of console wave `w`. That holds while each host wave takes
a run of consecutive ids: hosts tile the waves of a group of several rows
whose rows are 32 threads wide or more, or not a power of two wide
(measured on an RTX 3070 Ti and a Radeon iGPU; #2126 on an RTX 5070 Ti),
so such groups are refused; a group of one row keeps its runs in any
depth. The two host waves of a console wave are "halves". They are
scheduled independently: nothing orders one half's instruction against the
other's, except a group barrier.

The program is lowered in the per-lane form (`structure`): no EXEC, lane
masks are per-lane booleans, and a branch on a wave-wide mask test becomes
a branch on the lane's own bit. Scalar code runs in each half on its own;
a scalar is the same in both halves while everything it is computed from
is.

## What each operation needs

Level 1: exact within a half, nothing exchanged. Level 2: the halves
exchange values at a point every thread of the group reaches (below).
Level 3: an exchange inside a branch or a loop only some lanes run: where
only one half's lanes run it, that half runs it alone and nothing crosses
(`lower::confine`); elsewhere the region runs wave-wide, every thread
running every block of it, and a loop going round while any lane of the
group does (`lower::wide`), so the halves meet there.

| Operation | On the console (RDNA2, wave64) | In halves | Level |
|---|---|---|---|
| DPP16 (`quad_perm`, `row_shl/shr/ror`, mirrors, `row_share`, `row_xmask`) | within a row of 16 | exact; `row_mask` names the console's rows 0-3, so the row comes from the console lane | 1 |
| DPP8 | within 8 lanes | exact | 1 |
| `ds_swizzle_b32` | within 32 lanes (bit masks) or a quad | exact | 1 |
| `ds_bpermute_b32`, `ds_permute_b32` | within each 32-lane half ("as if two independent wave32's", ISA §10.4.4; measured in #573) | exact (`ds_permute_b32` is not lowered yet: #2152) | 1 |
| `v_permlane16_b32`, `v_permlanex16_b32` | within a row, or between the two rows of a half | exact (modelled only inside whole-wave sections today) | 1 |
| `v_mbcnt_lo/hi`, a lane's bit of a mask held as a number, `v_writelane_b32` | by the console lane | exact (console lane) | 1 |
| `v_readlane_b32` of a spilled SGPR | the SGPR | resolved in SSA (`Structured::restored`) | 1 |
| `v_readlane_b32` of a value the same in every lane of the console wave, written where every lane runs | that value | the lane's own | 1 |
| `v_readlane_b32`, any other | every lane gets lane L's value | the half holding lane L gives it (`Lower::across`) | 2, 3 |
| `v_readfirstlane_b32` of a value the same in every lane of the console wave (`halves::lane_uniform`), or a waterfall loop's descriptor read | that value | exact | 1 |
| `v_readfirstlane_b32`, any other | the first lane in EXEC of 64 | each half's first lane in EXEC and whether it has one, traded: the low half's if any (`Lower::first_lane`) | 2, 3 |
| A lane mask read as data (`s_bcnt1_i32_b64`, `s_ff1_i32_b64`, 64-bit compares, masks stored) | 64 bits | each half's ballot is one dword of it, traded (`Lower::mask_halves`) | 2, 3 |
| Whole-wave sections (reductions, prefix sums) | over 64 lanes | each half's part, traded and combined, low half first (`Lower::reduce_halves`) | 2, 3 |
| 16-bit whole-wave sums | over 64 lanes | refused | - |
| LDS in a group of one console wave (33 to 64 threads), no `s_barrier` | ordered by the wave running one instruction at a time | group barriers put the halves in that order: before an access that may meet an earlier one, and around a region only some lanes run (`halves::Lds`) | 2, 3 |
| LDS in a group of several console waves | ordered by `s_barrier` | the console's barriers | 1 |
| Any of the above in a group of 32 threads or fewer | no lane 32-63 exists | one host wave holds the whole console wave: nothing crosses (a ballot's high dword is 0, a lane read below 32 is the host's) | 1 |
| A group of several rows whose rows are 32 threads wide or more, or not a power of two wide | waves of consecutive thread ids | a host wave may hold a 16 x 2 tile of the group, so the relabelling would give two threads one id: refused | - |
| EXEC, branches on lane masks, loops with lane-dependent exits | EXEC masks | per-lane branches | 1 |
| `s_barrier` | per wave | group barrier | 1 |
| LDS, buffer and image atomics | per lane | per lane | 1 |
| GDS, `ds_ordered_count` | | refused on every host | - |

Every refusal names its reason, and a cross-half value no instruction ends
up needing costs nothing (the lowering retries with the value missing), so
a program is refused only when it uses one.

## Where the halves can meet

A Level 2 exchange (`Lower::trade`) is: each half writes its part to
group-shared slots, a group barrier (`dx.op.barrier`, SyncThreadGroup |
TGSMFence), each half reads both parts, and a second barrier before the
slots are written again. The barrier is what orders the two halves and
makes the writes visible; it must be reached by every thread of the group,
the same number of times:

- In D3D12 a barrier in non-uniform control flow is undefined. On NVIDIA
  since Volta a barrier waits for every thread of the group that has not
  exited, so a half that never arrives hangs the GPU (TDR).
- In the per-lane form a block is reached by every lane of the group only
  if no branch on a lane-dependent condition has it in its region (the
  blocks reachable from the branch that do not post-dominate it, the
  program's ends joined). A loop whose exit is lane-dependent is such a
  region, its latch included (each lane gets there once a round), and so
  is everything after a lane-dependent early end.
- In a group of one console wave, branches on uniform conditions are
  taken alike by both halves (their scalars are alike), so they are
  allowed. In a group of several console waves the other waves may take
  them differently: there, only points no branch encloses qualify.

Two ways of synchronizing are left out: waiting on a group-shared flag
(D3D12 guarantees no progress between waves of a group, and a half with no
lane at that point never writes it), and group barriers paired by count
from different places (#2214's review: undefined in D3D12).

Each half writes two dwords per exchange, beside the program's own LDS:
D3D12 gives a group 32 KiB of group-shared memory, the console up to 64 KiB
of LDS, so the slots must fit (else the program is refused).

## Level 3: regions only some lanes run

The per-lane form turns the console's wave-wide tests (`s_cbranch_vccz` on
a ballot, `s_cbranch_execz`) into per-lane branches: on the console the
whole wave enters such a region, here only the lanes whose bit is set, so a
half with none never gets there. Most exchanges the title needs sit in such
regions. Two cases are lowered, tried in this order after Level 2:

- **One half alone** (`lower::confine`). A block's entry condition
  (`Structured::reach`) is a decision diagram over atoms; compares of a
  lane's position (the console lane from `v_mbcnt`, the thread id of a
  group one row high, simple arithmetic on them) with constants are known
  in every lane. Where no lane of one half can meet the condition (`tid <
  16` in an LDS tree step), the other half's host wave runs the block
  alone, in the console's order: its ballot is the console's with the
  other dword 0, its first lane in EXEC and its reductions are the
  console's (a first lane under a guard that may leave EXEC empty is still
  refused), and LDS the region both writes and reads needs no barrier
  inside.
- **Wave-wide** (`lower::wide`). Otherwise the region runs wave-wide: the
  blocks not every thread meets at, joined by their edges, entered from
  one block and left to one, or holding the program's ends (or, where such
  a set is neither, the region of one branch in it). Every thread runs
  every block of it in order, each
  block after those its forward edges come from and a loop's blocks
  together, and each block acts only in its own lanes, those along its
  edges in. Each vector instruction's guard takes in the block's lanes (a
  VGPR write keeps the old value elsewhere, memory is accessed by those
  lanes only, ballots, first lanes and DPP source lanes see only them), the
  phis of the region and of its join pick, lane by lane, the value along
  the edge the lane came by, and every block of the region is one where
  every thread meets, so the halves trade and order LDS there. A lane read
  of a lane outside the block gives its old value, as on the console. A
  lane that gets to an end is in no block after it, and every thread ends
  together at the region's last block.
- A loop of such a region runs as the console's wave runs it: every
  thread goes round while any lane of the group does, which every half of
  the group tells the others at the end of each round (two barriers; in a
  group of several console waves, a wave whose lanes have all left goes
  round with none, which changes nothing it holds). The header's lanes are
  those entering, then those taking an edge back; a lane leaving along an
  edge out of the loop is out of it from then on, and after the loop the
  edge's lanes are all that left along it. What a lane holds stays as it
  left it: VGPR writes are guarded, and every other value a block of the
  loop defines (scalars, compare bits, what whole-wave sections leave,
  ballots) is kept per lane from the last round the lane ran the block, as
  each lane holds it in the per-lane form. Past the iteration limit
  (`LOOP_LIMIT`) the loop ends for every thread, and a lane still going
  round is in no block of the region after it.
- Only a region with no scalar code with effects beyond its values and
  loops entered at their headers runs wave-wide. Regions run wave-wide only
  where an
  attempt at lowering was refused for want of a meeting: a program that
  needs none is lowered as before, and one that does pays two barriers per
  exchange inside its regions (and per round of a loop) and a select per
  write there.

## Two console lanes per host lane

A console wave could also run in one host wave of 32 lanes, each host lane
holding console lanes `n` and `n + 32`: every VGPR value twice, scalars
once. Every cross-lane operation then stays inside one host wave (a
ballot is two host ballots, lane L is a read of the low or high copy), and
LDS of a one-wave group needs no barrier at all. It is not used here:

- every program in halves would pay twice the VGPRs and vector instructions
  per host lane, where wave-wide regions cost only the programs that cross
  inside a branch, and only there;
- the two console lanes of a host lane may branch differently, so the
  per-lane form no longer fits: control flow would follow EXEC (the console
  flow) for the whole program, a second lowering beside the per-lane one;
- what it gains over the halves, cross-lane reads inside loops with
  lane-dependent exits, wave-wide loops give within the per-lane form.

Running the two halves one after the other (lanes 0-31 to the end, then
32-63) is not exact for any program that reads across halves before its
end, and it reorders memory effects between them.

## Measured on the title

`victory translate --list --wave32 --validate` over the shader pack, the
executable, `XboxBvhBuilder.dll` and the FidelityFX upscaler DLL: wave64
compute programs that need console lanes run in halves. Compute programs
emitted, as each step was added (of 2,763; none lost on the way, every
program gained validates but for 6 in an existing validator failure class
that 735 compute programs share):

| Step | Compute programs emitted |
|---|---|
| #2633 (halves, Level 1) | 2,369 |
| groups of 32 threads or fewer | 2,375 |
| lane reads traded | 2,421 |
| LDS of one console wave put in order | 2,443 |
| first lanes, masks and reductions traded | 2,463 |
| regions only one half runs | 2,485 |
| wave-wide regions | 2,542 |
| wave-wide loops | 2,650 |
| regions to the program's end, loops of larger groups | 2,697 |

Still refused in halves: 9 first lanes, 6 one-wave LDS regions, 4 masks
and 2 reductions, in a region not entered from one block and left to one
(nor one of its branches' regions); 2 programs whose LDS leaves no room for
the trades' 16 bytes per console wave; 6 tiled group shapes and 2 16-bit
sums.

Before Levels 2 and 3, a local build that recorded every cross-half site
(and went on past each refusal, so every block was seen) classified the
415 wave64 programs in halves:

| Wave64 compute programs in halves | 415 |
|---|---|
| translate (no cross-half value used) | 41 |
| refused | 374 |
| - groups of 32 threads or fewer (no upper half) | 6 |
| - groups of one console wave, every exchange and LDS barrier where every lane meets | 108 |
| - groups of several console waves, no branch around any exchange | 5 |
| - an exchange, or an LDS write and read, inside a lane-dependent region | 247 |
| - other reasons | 8 |

By operation (refused programs that need it / of those, every site at a
point every lane reaches):

| `v_readfirstlane_b32` of values that may differ | 188 / 17 |
|---|---|
| A lane mask read as data | 164 / 16 |
| Whole-wave sections | 96 / 21 |
| `v_readlane_b32` | 54 / 50 |
| LDS of one console wave | 192 / 71 (no write and read in one lane-dependent region; 4 with every access at such a point) |

## Tests

`emit::halves_tests` runs hand-written programs on every hardware adapter,
in halves and as one wave64 where the adapter runs 64-lane waves, against
the console's results: lane reads, first lanes, masks and sums at meeting
points, in one-half regions and wave-wide, LDS of one-wave groups
(including a race that only barriers prevent), group shapes, and the
group-shared slots' cost in barriers; loops each lane leaves in its own
round (the halves going round different numbers of times) with first
lanes, masks and LDS inside, and a first lane read after such a loop.
`emit::halves_random_tests` runs random wave64 programs (cross-lane steps
at the top level, in branches one inside another and in loops each lane
leaves in its own round, LDS, scalar branches, lanes that end partway) in
halves and as one wave64 on an adapter that runs 64-lane waves, lane by
lane.

## Open pull requests in this area

- #223, #233, #320, #439, #2214 (stacked): the console flow, with
  exchanges between halves; LDS in one-wave groups; groups of several
  console waves.
- #189: Wave32 and Emulate modes, group-shared exchanges in uniform blocks,
  LDS fences at uniform points.
- #192, #2035: barriers for one-wave LDS; group sync for divergent
  barriers.
- #573: `ds_bpermute_b32` within halves. #2152: `ds_permute_b32`.
- #1890: split waves, exchanges between them.
- #2711: treats each half as its own wave (not exact where the console
  reads across halves).
