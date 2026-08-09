from pathlib import Path

from application.utils.logger import get_logger, setup_file_logger

from intent_classifier.config import IntentConfig
from intent_classifier.tokenizer import IntentTokenizer
from intent_classifier.predictor import IntentPredictor

logger = get_logger("intent.predict")

CHECKPOINT_DIR = Path("checkpoints/intent")
TRAIN_PATH = Path("data/intent/train.jsonl")


def main():
    config = IntentConfig()

    setup_file_logger(
        logger,
        log_dir=config.tensorboard_log_dir,
        filename="prediction.log",
    )

    logger.info("Starting intent predictor")

    tokenizer = IntentTokenizer()
    tokenizer.build_vocab_from_file(TRAIN_PATH)

    predictor = IntentPredictor(
        tokenizer=tokenizer,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    print("Intent Classifier Predictor")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text.lower() in ("quit", "exit"):
            break

        result = predictor.predict(text)

        print(
            f"Intent: {result.label} "
            f"(confidence: {result.confidence:.2%})\n"
        )

    logger.info("Predictor stopped")


if __name__ == "__main__":
    main()
