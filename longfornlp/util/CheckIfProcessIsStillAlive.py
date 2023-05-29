import os, errno
import psutil


def _is_running(pid):
    try:
        os.kill(pid, 0)  # 杀掉进程
    except:
        return False
    return True


def is_running(pname):
    for proc in psutil.process_iter():
        if proc.name() == pname:
            print(proc.pid)
            return True
    return False


if __name__ == "__main__":
    print(is_running("MobaXterm_Personal_12.3.exe"))
    print(_is_running(17352))
    print(is_running("MobaXterm_Personal_12.3.exe"))
