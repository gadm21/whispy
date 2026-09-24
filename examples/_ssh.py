"""Run shell commands on a Thoth node over SSH (paramiko).

Usage: python _ssh.py <host> "cmd1" "cmd2" ...
Defaults: user=gad password=password.
"""
import sys

import paramiko

USER = "gad"
PASS = "password"


def run(host: str, *cmds: str, timeout: int = 120) -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=USER, password=PASS, timeout=15)
    try:
        for cmd in cmds:
            print(f"--- {cmd}")
            stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
            out = stdout.read().decode()
            err = stderr.read().decode()
            if out:
                print(out)
            if err:
                print(err[:2000])
    finally:
        c.close()


if __name__ == "__main__":
    run(sys.argv[1], *sys.argv[2:])
