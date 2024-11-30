from scipy.stats import spearmanr

def calculate_spearman(gold_standard, rankings):
    """
    Calculate the Spearman correlation for each ranking compared to the gold standard.
    :param gold_standard: List representing the gold standard ranking (e.g., [1, 2, 3, 4])
    :param rankings: List of rankings
    :return: List of Spearman correlation coefficients
    """
    def rank_to_position(rank):
        return [rank.index(i) + 1 for i in range(1, len(rank) + 1)]

    gold_positions = rank_to_position(gold_standard)
    correlations = []

    for ranking in rankings:
        positions = rank_to_position(ranking)
        correlation, _ = spearmanr(gold_positions, positions)
        correlations.append(correlation)
    
    return correlations

gold_standard = [2, 3, 1, 4]
rankings = [
    [2, 3, 1, 4], 
    [3, 1, 2, 4], 
    [4, 3, 2, 1], 
    [1, 2, 3, 4], 
    [3, 4, 1, 2], 
    [2, 1, 4, 3], 
    [4, 1, 3, 2], 
]

correlations = calculate_spearman(gold_standard, rankings)

for i, correlation in enumerate(correlations, start=1):
    print(f"{i}: Spearman Correlation = {correlation:.3f}")
