from collections import defaultdict
import math

def train_hmm_with_separate_inputs(tokens, pos_tags, ner_tags, alpha=1.0):
    if not (len(tokens) == len(pos_tags) == len(ner_tags)):
        raise ValueError("Input lists (tokens, pos_tags, ner_tags) must have the same length.")

    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    label_counts = defaultdict(int)
    vocab = set()
    label_set = set()  # To store unique labels seen during training

    # Iterate over sentences
    for sentence_tokens, sentence_pos_tags, sentence_ner_tags in zip(tokens, pos_tags, ner_tags):
        prev_label = None
        for word, pos_tag, ner_tag in zip(sentence_tokens, sentence_pos_tags, sentence_ner_tags):
            label_counts[ner_tag] += 1
            emission_counts[ner_tag][(word, pos_tag)] += 1
            vocab.add((word, pos_tag))
            label_set.add(ner_tag)  # Add NER tag to the label set
            if prev_label is not None:
                transition_counts[prev_label][ner_tag] += 1
            prev_label = ner_tag

    vocab_size = len(vocab)

    # Calculate transition probabilities with smoothing
    transition_probs = {
        label: {next_label: math.log((count + alpha) /
                                     (sum(next_counts.values()) + alpha * len(label_set)))
                for next_label, count in next_counts.items()}
        for label, next_counts in transition_counts.items()
    }

    # Calculate emission probabilities with smoothing
    emission_probs = {
        label: {word_pos: math.log((count + alpha) /
                                   (label_counts[label] + alpha * vocab_size))
                for word_pos, count in word_counts.items()}
        for label, word_counts in emission_counts.items()
    }

    # Add smoothing for unseen words
    for label in label_counts:
        emission_probs[label]["<UNK>"] = math.log(alpha / (label_counts[label] + alpha * vocab_size))

    # Calculate label priors
    label_priors = {
        label: math.log(count / sum(label_counts.values()))
        for label, count in label_counts.items()
    }

    return transition_probs, emission_probs, label_priors, vocab, label_set


def viterbi_with_tokens_and_pos_tags(sequence_with_tags, transition_probs, emission_probs, label_priors, vocab, label_set):
    n = len(sequence_with_tags)
    dp = [{} for _ in range(n)]
    backpointers = [{} for _ in range(n)]

    # Replace unseen (word, POS) pairs with <UNK>
    sequence_with_tags = [(word_tag if word_tag in vocab else "<UNK>") for word_tag in sequence_with_tags]

    # Initialize first token
    for label in label_set:
        dp[0][label] = label_priors.get(label, -float('inf')) + \
                       emission_probs[label].get(sequence_with_tags[0], -float('inf'))
        backpointers[0][label] = None

    # Dynamic programming for Viterbi
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

    # Backtrack to find the best sequence
    best_sequence = []
    best_label = max(dp[-1], key=dp[-1].get)
    best_sequence.append(best_label)
    for t in range(n-1, 0, -1):
        best_label = backpointers[t][best_label]
        best_sequence.append(best_label)

    best_sequence.reverse()
    return best_sequence
