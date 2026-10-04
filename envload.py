"""env.txt (또는 .env) 를 읽어 환경변수로 넣는다. 이미 설정된 값은 덮어쓰지 않는다."""
import os, pathlib

for name in ("env.txt", ".env"):
    p = pathlib.Path(__file__).parent / name
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v and not os.environ.get(k):
                os.environ[k] = v
        break
