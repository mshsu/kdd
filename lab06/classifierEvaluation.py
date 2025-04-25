import pandas as pd
import sys


def main():
    args = sys.argv
    if len(args) < 2:
        print("Syntax: python3 classifierEvaluation.py <resultsFile>")
    else:
        filename = args[1]
        results = pd.read_csv(filename)

        accuracy = results.copy()
        accuracy['match'] = (accuracy['obs'] == accuracy['pred']) * 1
        accuracy['nomatch'] = accuracy['match'] * -1 + 1

        hits = accuracy.groupby('obs')['match'].sum().sort_values(
            ascending=False)
        strikes = accuracy.groupby('pred')['nomatch'].sum().sort_values(
            ascending=False)
        misses = accuracy.groupby('obs')['nomatch'].sum().sort_values(
            ascending=False)

        author_summary = pd.concat([hits, strikes, misses], axis=1)
        author_summary = author_summary.reset_index()
        author_summary.columns = ['Author', 'Hits', 'Strikes', 'Misses']
        author_summary['Precision'] = author_summary['Hits'] / (
                    author_summary['Hits'] + author_summary['Strikes'])
        author_summary['Recall'] = author_summary['Hits'] / (
                    author_summary['Hits'] + author_summary['Misses'])
        author_summary['F1'] = 2 / (
                    1 / author_summary['Precision'] + 1 / author_summary[
                'Recall'])

        overall_accuracy = hits.sum() / (hits.sum() + misses.sum())
        total_correct = int(hits.sum())
        total_incorrect = int(misses.sum())
        conf_matrix = pd.crosstab(accuracy['obs'], accuracy['pred'])

        print("Results Summary:")
        print(f"File: {filename.split('/')[-1]}")
        print()
        print("By Author:")
        print(author_summary.to_string(index=False))
        print()
        print("Overall:")
        print(f"N Correctly Classified:   {total_correct}")
        print(f"N Incorrectly Classified: {total_incorrect}")
        print(f"Accuracy:                 {overall_accuracy:0.3f}")
        print()
        print("Confusion Matrix:")
        print(conf_matrix.to_string())


if __name__ == "__main__":
    main()