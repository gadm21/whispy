"""Run shell commands on thoth-chen over SSH (paramiko)."""
import sys
import paramiko

HOST = "10.0.0.22"  # thoth-chen.local — mDNS flakes under load, use IP
USER = "gad"
PASS = "password"


def run(*cmds: str, timeout: int = 60) -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)
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
    run(*sys.argv[1:])
