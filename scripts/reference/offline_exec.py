"""Run a prepared reference command with kernel-enforced network denial on Linux.

Install dependencies and download checkpoints separately. This launcher fails
closed if libseccomp or filter installation is unavailable. The filter is
inherited across exec and by child processes; environment flags alone are not
treated as a network boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import os
import stat
import sys


def deny_network() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Kernel-enforced offline execution requires Linux and libseccomp")
    path = ctypes.util.find_library("seccomp")
    if path is None:
        raise RuntimeError("libseccomp is unavailable; refusing to execute the reference command")
    library = ctypes.CDLL(path, use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("Could not allocate the offline seccomp policy")
    try:
        # Also deny connect/send operations on any inherited socket descriptor.
        # No socket families are exempted, including Unix-domain sockets.
        for name in ("socket", "socketpair", "connect", "sendto", "sendmsg", "sendmmsg",
                     "io_uring_setup", "io_uring_enter", "io_uring_register"):
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                raise RuntimeError(f"Cannot resolve network syscall: {name}")
            result = library.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0)
            if result != 0:
                raise RuntimeError(f"Could not deny {name}: errno {-result}")
        result = library.seccomp_load(context)
        if result != 0:
            raise RuntimeError(f"Could not install offline policy: errno {-result}")
    finally:
        library.seccomp_release(context)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Executable and arguments, after --; no shell expansion")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a prepared reference command is required")
    try:
        for descriptor in range(3):
            if stat.S_ISSOCK(os.fstat(descriptor).st_mode):
                raise RuntimeError("Standard input/output must not be inherited sockets")
        deny_network()
        # No already-connected descriptors or io_uring handles enter the child.
        os.closerange(3, os.sysconf("SC_OPEN_MAX"))
        os.execvpe(command[0], command, os.environ)
    except (OSError, RuntimeError) as error:
        print(f"Offline execution refused: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
