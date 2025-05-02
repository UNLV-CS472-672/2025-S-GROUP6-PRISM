from cheating.models import SubmissionSimilarityPairs

def get_all_similarity_pairs(course_id, semester_id):
    """
    Fetach all similarity pairs information  given a course and a semester.
    """

    # retrieveing all similarity pairs from the database
    # Moss reports enforces unique pairs so we dont need to worry about duplicates
    # such as (A, B) and (B, A)
    # this tho, will return M duplicates of pairs (A, B), where M is the number of assignments
    similarity_pairs_qs = SubmissionSimilarityPairs.objects.filter(
        assignment__course_catalog_id=course_id,
        assignment__semester_id=semester_id,
        ).values(
            'submission_id_1__student_id',
            'submission_id_2__student_id',
            'percentage',
            'assignment_id'
        )

    # making a python list of dictionaries from the queryset
    similarity_pairs: list[dict] = list(similarity_pairs_qs)

    return similarity_pairs

def turn_pairs_to_dict(similarity_pairs):
    """
    Convert a list of similarity pairs into a dictionary.

    The dictionary will use pairs of student IDs as keys
    and will store their similarity scores and assignment IDs as values.
    """

    # A dictionary that takes pairs of student IDs as keys
    # and returns a list with their similarity scores for each assignment
    # and the assignment ID
    pairs_dict: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for pair in similarity_pairs:
        student_id_1 = pair['submission_id_1__student_id']
        student_id_2 = pair['submission_id_2__student_id']
        similarity_score = pair['percentage']
        assignment_id = pair['assignment_id']

        # Create a sorted tuple of student IDs to ensure uniqueness
        # here (A, B) is the same as (B, A) because example
        #(10, 20) and (20, 10) after sorting they both become (10, 20)
        key = tuple(sorted((student_id_1, student_id_2)))

        # Initialize the empty list if the key does not exist
        if key not in pairs_dict:
            pairs_dict[key] = []

        # Append the similarity score and assignment ID as a tuple
        pairs_dict[key].append((similarity_score, assignment_id))

    return pairs_dict

def compute_pairs_statistics(
    pairs_dict: dict[tuple[int,int], list[tuple[int,int]]]
) -> tuple[dict[tuple[int,int], dict], dict]:
    """
    Compute statistics from the similarity pairs dictionary.

    Args:
        pairs_dict: A dict mapping each student‐pair (tuple of IDs) to a list of
            (similarity_score, assignment_id) tuples.

    Returns:
        per_pair_stats: A dict where each key is a student‐pair and each value is a
            dict containing:
            {
                'total_assignments': int,
                'average_similarity_score': float,
                'max_similarity_score': float,
                'min_similarity_score': float,
                'mean_z_score': float,
                'max_z_score': float,
                'min_z_score': float
            }
        population_stats: A dict containing:
            {
                'total_pairs': int,
                'population_mean': float,
                'population_variance': float,
                'population_standard_deviation': float
            }
    """

    def compute_pair_z_score(pair_stat, mean, std_dev):
        """
        Calculate the mean z-score for a student pair's.

        This function computes the mean z-score, maximum z-score, and minimum z-score
        for a student pair's similarity scores
        based on the provided population mean and standard deviation.
        """
        # guard against division by zero
        if std_dev == 0:
            return 0, 0, 0

        mean_z_score = (pair_stat['average_similarity_score'] - mean) / std_dev
        max_z_score = (pair_stat['max_similarity_score'] - mean) / std_dev
        min_z_score = (pair_stat['min_similarity_score'] - mean) / std_dev

        return mean_z_score, max_z_score, min_z_score


    statistics_dict: dict[tuple[int, int], dict] = {}
    total_pairs = len(pairs_dict)
    population_similarity_score_sum = 0

    for pair, scores in pairs_dict.items():
        total_assignments = len(scores)
        if total_assignments == 0:
            continue

        # Extract similarity scores from the list of tuples
        similarity_scores = [score[0] for score in scores]

        # Calculate average, max, and min similarity scores
        average_similarity_score = sum(similarity_scores) / total_assignments
        max_similarity_score = max(similarity_scores)
        min_similarity_score = min(similarity_scores)

        # Store the statistics in a dictionary
        statistics_dict[pair] = {
            'total_assignments': total_assignments,
            'average_similarity_score': average_similarity_score,
            'max_similarity_score': max_similarity_score,
            'min_similarity_score': min_similarity_score,
        }

        # Update the population similarity score sum
        population_similarity_score_sum += average_similarity_score

    mu = population_similarity_score_sum / total_pairs if total_pairs > 0 else 0
    population_deviation_sum = 0

    for pair, stats in statistics_dict.items():
        average_similarity_score = stats['average_similarity_score']
        # Calculate the deviation from the population mean
        deviation = average_similarity_score - mu
        population_deviation_sum += deviation ** 2

    population_variance = population_deviation_sum / total_pairs if total_pairs > 0 else 0
    population_standard_deviation = population_variance ** 0.5

    # Create a dictionary to store population statistics
    population_stat = {
        'total_pairs': total_pairs,
        'population_mean': mu,
        'population_variance': population_variance,
        'population_standard_deviation': population_standard_deviation,
    }

    # Calculate z-scores for each pair
    for pair, stats in statistics_dict.items():
        pair_z_scores_info = compute_pair_z_score(
            stats,
            mu,
            population_standard_deviation
        )
        statistics_dict[pair]['mean_z_score'] = pair_z_scores_info[0]
        statistics_dict[pair]['max_z_score'] = pair_z_scores_info[1]
        statistics_dict[pair]['min_z_score'] = pair_z_scores_info[2]


    return (statistics_dict, population_stat)

