# Copyright (c) 2023 The University of Edinburgh
# Copyright (c) 2025 Technical University of Munich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
GAPBS on Neoverse V2 in SE mode

Runs GAP Benchmark Suite kernels on the Neoverse V2 O3 CPU model with
L1I/L1D/L2 caches and optional Fetch Directed Prefetcher (FDP).

Usage
-----

```
scons build/ARM/gem5.opt
./build/ARM/gem5.opt configs/example/arm/gapbs_neoverse_v2.py \
    --benchmark bfs --binary-dir /path/to/gapbs --graph-scale 10 --num-trials 1
```
"""

import argparse
import os

import m5
from m5.util import addToPath

m5.util.addToPath("../..")

from common.cores.arm import neoverse_v2

from m5.objects import (
    FetchDirectedPrefetcher,
    L2XBar,
    MultiPrefetcher,
    TaggedPrefetcher,
)

from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.caches.mmu_cache import MMUCache
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires

BENCHMARKS = ["bc", "bfs", "cc", "cc_sv", "pr", "pr_spmv", "sssp", "tc"]

parser = argparse.ArgumentParser(
    description="Run GAPBS benchmarks on Neoverse V2 in SE mode."
)

parser.add_argument(
    "--benchmark",
    type=str,
    required=True,
    choices=BENCHMARKS,
    help="GAPBS kernel to run.",
)

parser.add_argument(
    "--binary-dir",
    type=str,
    required=True,
    help="Path to directory containing GAPBS binaries.",
)

parser.add_argument(
    "--graph-scale",
    type=int,
    default=10,
    help="Generate synthetic Kronecker graph with 2^scale vertices (default: 10).",
)

parser.add_argument(
    "--graph-file",
    type=str,
    default=None,
    help="Load graph from file instead of generating synthetic graph.",
)

parser.add_argument(
    "--num-trials",
    type=int,
    default=1,
    help="Number of benchmark trials (default: 1).",
)

parser.add_argument(
    "--disable-fdp",
    action="store_true",
    help="Disable FDP to evaluate baseline performance.",
)

parser.add_argument(
    "--memory-size",
    type=str,
    default="8192MiB",
    help="Memory size (default: 8192MiB).",
)

args = parser.parse_args()

requires(isa_required=ISA.ARM)

memory = SingleChannelDDR3_1600(size=args.memory_size)


class CacheHierarchy(PrivateL1PrivateL2CacheHierarchy):
    def __init__(self):
        super().__init__("", "", "")

    def incorporate_cache(self, board: AbstractBoard) -> None:
        board.connect_system_port(self.membus.cpu_side_ports)

        for _, port in board.get_memory().get_mem_ports():
            self.membus.mem_side_ports = port

        self.l1icaches = [
            neoverse_v2.L1I()
            for i in range(board.get_processor().get_num_cores())
        ]

        for i in range(board.get_processor().get_num_cores()):
            cpu = board.get_processor().cores[i].core

            self.l1icaches[i].prefetcher = MultiPrefetcher()
            if not args.disable_fdp:
                pf = FetchDirectedPrefetcher(
                    use_virtual_addresses=True, cpu=cpu
                )
                pf.registerCache(self.l1icaches[i])
                self.l1icaches[i].prefetcher.prefetchers.append(pf)

            self.l1icaches[i].prefetcher.prefetchers.append(
                TaggedPrefetcher(use_virtual_addresses=True)
            )

            for pf in self.l1icaches[i].prefetcher.prefetchers:
                pf.registerMMU(cpu.mmu)

        self.l1dcaches = [
            neoverse_v2.L1D()
            for i in range(board.get_processor().get_num_cores())
        ]
        self.l2buses = [
            L2XBar() for i in range(board.get_processor().get_num_cores())
        ]
        self.l2caches = [
            neoverse_v2.L2()
            for i in range(board.get_processor().get_num_cores())
        ]
        self.mmucaches = [
            MMUCache(size="8KiB")
            for _ in range(board.get_processor().get_num_cores())
        ]

        self.mmubuses = [
            L2XBar(width=64)
            for i in range(board.get_processor().get_num_cores())
        ]

        if board.has_coherent_io():
            self._setup_io_cache(board)

        for i, cpu in enumerate(board.get_processor().get_cores()):

            cpu.connect_icache(self.l1icaches[i].cpu_side)
            self.l1icaches[i].mem_side = self.l2buses[i].cpu_side_ports

            cpu.connect_dcache(self.l1dcaches[i].cpu_side)
            self.l1dcaches[i].mem_side = self.l2buses[i].cpu_side_ports

            self.mmucaches[i].mem_side = self.l2buses[i].cpu_side_ports

            self.mmubuses[i].mem_side_ports = self.mmucaches[i].cpu_side
            self.l2buses[i].mem_side_ports = self.l2caches[i].cpu_side

            self.membus.cpu_side_ports = self.l2caches[i].mem_side

            cpu.connect_walker_ports(
                self.mmubuses[i].cpu_side_ports,
                self.mmubuses[i].cpu_side_ports,
            )

            cpu.connect_interrupt()


cache_hierarchy = CacheHierarchy()

processor = BaseCPUProcessor(
    cores=[BaseCPUCore(neoverse_v2.NeoverseV2(), isa=ISA.ARM)]
)

for core in processor.cores:
    cpu = core.core
    if args.disable_fdp:
        cpu.decoupledFrontEnd = False
    else:
        cpu.decoupledFrontEnd = True

# Resolve binary path
binary_path = os.path.abspath(os.path.join(args.binary_dir, args.benchmark))
if not os.path.isfile(binary_path):
    raise FileNotFoundError(
        f"GAPBS binary not found: {binary_path}\n"
        f"Build GAPBS with: cd gapbs && CXX=aarch64-linux-gnu-g++ SERIAL=1 make"
    )

# Build arguments for the GAPBS binary
bench_args = []
if args.graph_file:
    bench_args.extend(["-f", args.graph_file])
else:
    bench_args.extend(["-g", str(args.graph_scale)])
bench_args.extend(["-n", str(args.num_trials)])

print(
    f"Running GAPBS {args.benchmark} on NeoverseV2 "
    f"FDP {'disabled' if args.disable_fdp else 'enabled'} "
    f"args: {' '.join(bench_args)}"
)

board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

binary = BinaryResource(local_path=binary_path)
board.set_se_binary_workload(binary, arguments=bench_args)

simulator = Simulator(board=board)
simulator.run()

print("Simulation done.")
