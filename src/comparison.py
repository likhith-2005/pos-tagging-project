import sys
import os
sys.path.append(os.path.dirname(__file__))

from analysis import models, evaluate


def print_header():
    print("\n" + "="*50)
    print("        📊 POS TAGGING MODEL COMPARISON")
    print("="*50)


def print_table(results):
    print("\n📋 PERFORMANCE TABLE:\n")

    print("{:<5} {:<15} {:<15} {:<15}".format("Rank", "Model", "Accuracy", "Percentage"))
    print("-"*55)

    # Sort models by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (model, acc) in enumerate(sorted_results, start=1):
        print("{:<5} {:<15} {:<15} {:<15}".format(
            i,
            model,
            f"{acc:.4f}",
            f"{acc*100:.2f}%"
        ))

    return sorted_results


def print_summary(sorted_results):
    best_model, best_acc = sorted_results[0]
    worst_model, worst_acc = sorted_results[-1]

    print("\n" + "="*50)
    print("📌 SUMMARY")
    print("="*50)

    print(f"🏆 Best Model     : {best_model} ({best_acc:.4f})")
    print(f"⚠️  Worst Model   : {worst_model} ({worst_acc:.4f})")

    print("\n📊 Accuracy Gap:")
    print(f"{best_model} outperforms {worst_model} by {(best_acc - worst_acc)*100:.2f}%")

    print("="*50 + "\n")


def main():
    print_header()

    results = {}

    for name, model in models.items():
        try:
            print(f"🔄 Evaluating {name}...")
            acc = evaluate(model)
            results[name] = acc
        except Exception as e:
            print(f"❌ {name} failed: {e}")

    sorted_results = print_table(results)
    print_summary(sorted_results)


if __name__ == "__main__":
    main()