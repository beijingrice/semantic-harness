"""命令行版：bilingual-check --en '...' --zh '...'，亦可读取 UTF-8 文件。"""

import argparse
import json
import sys
from pathlib import Path

from app.backends import BackendError
from app.checker import Checker
from app.config import ConfigError, Settings
from app.llm import ChatClient, LLMError


def main() -> None:
    parser = argparse.ArgumentParser(description="本地中英双向关键词完整性检测")
    en = parser.add_mutually_exclusive_group(required=True)
    en.add_argument("--en", help="英文文本")
    en.add_argument("--en-file", type=Path, help="UTF-8 英文文本文件")
    zh = parser.add_mutually_exclusive_group(required=True)
    zh.add_argument("--zh", help="中文文本")
    zh.add_argument("--zh-file", type=Path, help="UTF-8 中文文本文件")
    parser.add_argument("--model", help="覆盖 .env 中的模型 ID")
    parser.add_argument("--verifier-model", help="覆盖 LM_VERIFIER_MODEL，指定第二个复核模型")
    parser.add_argument("--mode", choices=("primary", "dual"), default="dual", help="初判基线或双模型+仲裁")
    args = parser.parse_args()
    try:
        english = args.en if args.en is not None else args.en_file.read_text(encoding="utf-8")
        chinese = args.zh if args.zh is not None else args.zh_file.read_text(encoding="utf-8")
        if not english.strip() or not chinese.strip():
            parser.error("中英文文本都不能为空")
        settings = Settings()
        client = ChatClient(settings)
        try:
            result = Checker(client, settings).check(english, chinese, args.model, args.verifier_model, args.mode)
        finally:
            client.close()
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        sys.exit(0 if result.verdict == "PASS" else 3 if result.verdict == "REVIEW_REQUIRED" else 1)
    except (ConfigError, BackendError) as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except (OSError, LLMError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
