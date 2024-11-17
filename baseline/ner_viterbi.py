from collections import defaultdict
import math

def train_hmm_with_pos_tags(training_data, alpha=1.0):
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    label_counts = defaultdict(int)
    vocab = set()

    for sentence, labels in training_data:
        prev_label = None
        for (word, tag), label in zip(sentence, labels):
            label_counts[label] += 1
            emission_counts[label][(word, tag)] += 1
            vocab.add((word, tag))
            if prev_label is not None:
                transition_counts[prev_label][label] += 1
            prev_label = label

    vocab_size = len(vocab)

    transition_probs = {
        label: {next_label: math.log((count + alpha) /
                                     (sum(next_counts.values()) + alpha * len(label_counts)))
                for next_label, count in next_counts.items()}
        for label, next_counts in transition_counts.items()
    }

    emission_probs = {
        label: {word_tag: math.log((count + alpha) /
                                   (label_counts[label] + alpha * vocab_size))
                for word_tag, count in word_counts.items()}
        for label, word_counts in emission_counts.items()
    }

    for label in label_counts:
        emission_probs[label]["<UNK>"] = math.log(alpha / (label_counts[label] + alpha * vocab_size))

    label_priors = {
        label: math.log(count / sum(label_counts.values()))
        for label, count in label_counts.items()
    }

    return transition_probs, emission_probs, label_priors, vocab

def viterbi_with_pos_tags(sequence_with_tags, transition_probs, emission_probs, label_priors, label_set, vocab):
    n = len(sequence_with_tags)
    dp = [{} for _ in range(n)]
    backpointers = [{} for _ in range(n)]

    sequence_with_tags = [(word_tag if word_tag in vocab else "<UNK>") for word_tag in sequence_with_tags]

    for label in label_set:
        dp[0][label] = label_priors.get(label, -float('inf')) + \
                       emission_probs[label].get(sequence_with_tags[0], -float('inf'))
        backpointers[0][label] = None

    for t in range(1, n):
        for label in label_set:
            max_prob, best_label = max(
                (dp[t-1][prev_label] +
                 transition_probs[prev_label].get(label, -float('inf')) +
                 emission_probs[label].get(sequence_with_tags[t], -float('inf')),
                 prev_label)
                for prev_label in label_set
            )
            dp[t][label] = max_prob
            backpointers[t][label] = best_label

    best_sequence = []
    best_label = max(dp[-1], key=dp[-1].get)
    best_sequence.append(best_label)
    for t in range(n-1, 0, -1):
        best_label = backpointers[t][best_label]
        best_sequence.append(best_label)

    best_sequence.reverse()
    return best_sequence
