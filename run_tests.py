import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / "tests"
PYTHON = BASE_DIR / ".venv" / "bin" / "python"


def run_pytest(args: list[str]) -> int:
    cmd = [str(PYTHON), "-m", "pytest"] + args
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode


def menu() -> None:
    options = {
        "1": ("Run all unit tests", ["tests/unit/", "-v", "-m", "not integration"]),
        "2": ("Run all integration tests", ["tests/integration/", "-v", "-m", "integration"]),
        "3": ("Run all tests", ["tests/", "-v"]),
        "4": ("Run RAG unit tests", ["tests/unit/rag/", "-v"]),
        "5": ("Run Chat unit tests", ["tests/unit/chat/", "-v"]),
        "6": ("Run Intent Classifier unit tests", ["tests/unit/intent_classifier/", "-v"]),
        "7": ("Run Bootstrap unit tests", ["tests/unit/bootstrap/", "-v"]),
        "8": ("Run specific test file", None),
        "9": ("Exit", None),
    }

    while True:
        print("\n" + "=" * 50)
        print("TEST RUNNER")
        print("=" * 50)
        for key, (label, _) in options.items():
            print(f"  {key}. {label}")
        print("=" * 50)

        choice = input("Select an option: ").strip()

        if choice not in options:
            print("Invalid option. Please try again.")
            continue

        label, args = options[choice]

        if choice == "9":
            print("Exiting...")
            sys.exit(0)

        if choice == "8":
            test_path = input("Enter test file path (relative to project root): ").strip()
            if not test_path:
                print("No path provided.")
                continue
            full_path = BASE_DIR / test_path
            if not full_path.exists():
                print(f"File not found: {test_path}")
                continue
            returncode = run_pytest([test_path, "-v"])
        else:
            returncode = run_pytest(args)

        if returncode == 0:
            print("\n✅ Tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code {returncode}")


if __name__ == "__main__":
    menu()
