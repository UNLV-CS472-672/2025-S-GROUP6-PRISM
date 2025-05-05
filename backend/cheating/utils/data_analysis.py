"""Utility functions for analyzing student submission similarity data."""

import logging
from sklearn.cluster import KMeans
import numpy as np
import time
from django.db import transaction
from cheating.models import (
    SubmissionSimilarityPairs,
    StudentPairSimilarityStatistics,
    PopulationSimilarityStatistics,
    PairFlagStat,
)
from courses.models import CourseCatalog, Semester, Students

logger = logging.getLogger(__name__)


def get_all_similarity_pairs(course_id, semester_id):
    """Fetach all similarity pairs information  given a course and a semester."""
    # retrieveing all similarity pairs from the database
    # Moss reports enforces unique pairs so we dont need to worry about duplicates
    # such as (A, B) and (B, A)
    # this tho, will return M duplicates of pairs (A, B), where M is the number of assignments
    similarity_pairs_qs = SubmissionSimilarityPairs.objects.filter(
        assignment__course_catalog_id=course_id,
        assignment__semester_id=semester_id,
    ).values(
        "submission_id_1__student_id",
        "submission_id_2__student_id",
        "percentage",
        "assignment_id",
    )

    # making a python list of dictionaries from the queryset
    similarity_pairs: list[dict] = list(similarity_pairs_qs)

    return similarity_pairs


def turn_pairs_to_dict(similarity_pairs):
    """
    Convert a list of similarity pairs into a dictionary.

    The dictionary will use pairs of student IDs as keys
    and will store their similarity scores and assignment IDs as values.
    the dictionary will have the following structure:
    {
        (student_id_1, student_id_2): [
            (similarity_score_1, assignment_id_1),
            (similarity_score_2, assignment_id_2),
            ...
        ],
        ...
    }
    """
    # A dictionary that takes pairs of student IDs as keys
    # and returns a list with their similarity scores for each assignment
    # and the assignment ID
    pairs_dict: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for pair in similarity_pairs:
        student_id_1 = pair["submission_id_1__student_id"]
        student_id_2 = pair["submission_id_2__student_id"]
        similarity_score = pair["percentage"]
        assignment_id = pair["assignment_id"]

        # Create a sorted tuple of student IDs to ensure uniqueness
        # here (A, B) is the same as (B, A) because example
        # (10, 20) and (20, 10) after sorting they both become (10, 20)
        key = tuple(sorted((student_id_1, student_id_2)))

        # Initialize the empty list if the key does not exist
        if key not in pairs_dict:
            pairs_dict[key] = []

        # Append the similarity score and assignment ID as a tuple
        pairs_dict[key].append((similarity_score, assignment_id))

    return pairs_dict


def compute_pairs_statistics(
    pairs_dict: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], dict], dict]:
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
                'total_similarity_score': float,
                'median_similarity_score': float,
                'sample_standard_deviation': float,
                'sample_variance': float,
                'mean_z_score': float,
                'max_z_score': float,
                'min_z_score': float,
                'total_z_score': float,
                'flagged': bool
                }

        population_stat: A dict containing overall statistics for the population of student pairs:
            {
                'total_pairs': int,
                'population_mean': float,
                'population_variance': float,
                'population_standard_deviation': float,
                'population_median_similarity_score': float,
                'population_median_z_score': float
                'population_mean_z_score': float
            }
    """

    def compute_pair_z_score_info(pair_stat, mean, std_dev):
        """
        Calculate the mean z-score for a student pair's.

        This function computes the mean z-score, maximum z-score, and minimum z-score
        for a student pair's similarity scores
        based on the provided population mean and standard deviation.
        """
        # guard against division by zero
        if std_dev == 0:
            return 0, 0, 0

        mean_z_score = (pair_stat["average_similarity_score"] - mean) / std_dev
        max_z_score = (pair_stat["max_similarity_score"] - mean) / std_dev
        min_z_score = (pair_stat["min_similarity_score"] - mean) / std_dev

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
        total_similarity_score = sum(similarity_scores)
        average_similarity_score = total_similarity_score / total_assignments
        max_similarity_score = max(similarity_scores)
        min_similarity_score = min(similarity_scores)

        sample_deviation_sum = 0
        for score in similarity_scores:
            sample_deviation_sum += (score - average_similarity_score) ** 2
        sample_variance = (
            sample_deviation_sum / (total_assignments - 1)
            if total_assignments > 1
            else 0
        )
        sample_standard_deviation = sample_variance**0.5

        # find the median similarity score (what lies in the middle of the sorted list)
        similarity_scores.sort()
        if total_assignments % 2 == 0:
            median_similarity_score = (
                similarity_scores[total_assignments // 2 - 1]
                + similarity_scores[total_assignments // 2]
            ) / 2
        else:
            median_similarity_score = similarity_scores[total_assignments // 2]

        # Store the statistics in a dictionary
        statistics_dict[pair] = {
            "total_assignments": total_assignments,
            "average_similarity_score": average_similarity_score,
            "max_similarity_score": max_similarity_score,
            "min_similarity_score": min_similarity_score,
            "total_similarity_score": total_similarity_score,
            "median_similarity_score": median_similarity_score,
            "sample_standard_deviation": sample_standard_deviation,
            "sample_variance": sample_variance,
        }

        # Update the population similarity score sum
        population_similarity_score_sum += average_similarity_score

    mu = population_similarity_score_sum / total_pairs if total_pairs > 0 else 0
    population_deviation_sum = 0

    for pair, stats in statistics_dict.items():
        average_similarity_score = stats["average_similarity_score"]
        # Calculate the deviation from the population mean
        deviation = average_similarity_score - mu
        population_deviation_sum += deviation**2

    population_variance = (
        population_deviation_sum / total_pairs if total_pairs > 0 else 0
    )
    population_standard_deviation = population_variance**0.5

    similarity_score_list = [
        stats["average_similarity_score"] for stats in statistics_dict.values()
    ]
    similarity_score_list.sort()

    # find the median similarity score for the population
    if total_pairs % 2 == 0:
        median_similarity_score_population = (
            similarity_score_list[total_pairs // 2 - 1]
            + similarity_score_list[total_pairs // 2]
        ) / 2
    else:
        median_similarity_score_population = similarity_score_list[total_pairs // 2]

    # Create a dictionary to store population statistics
    population_stat = {
        "total_pairs": total_pairs,
        "population_mean": mu,
        "population_variance": population_variance,
        "population_standard_deviation": population_standard_deviation,
        "population_median_similarity_score": median_similarity_score_population,
    }

    # Calculate z-scores for each pair
    for pair, stats in statistics_dict.items():
        pair_z_scores_info = compute_pair_z_score_info(
            stats, mu, population_standard_deviation
        )
        statistics_dict[pair]["mean_z_score"] = pair_z_scores_info[0]
        statistics_dict[pair]["max_z_score"] = pair_z_scores_info[1]
        statistics_dict[pair]["min_z_score"] = pair_z_scores_info[2]
        if pair_z_scores_info[0] >= 2:
            statistics_dict[pair]["flagged"] = True
        else:
            statistics_dict[pair]["flagged"] = False

    z_score_list = [stats["mean_z_score"] for stats in statistics_dict.values()]
    population_mean_z_score = sum(z_score_list) / total_pairs if total_pairs > 0 else 0
    population_stat["population_mean_z_score"] = population_mean_z_score

    z_score_list.sort()
    # find the median z-score for the population
    if total_pairs % 2 == 0:
        median_z_score_population = (
            z_score_list[total_pairs // 2 - 1] + z_score_list[total_pairs // 2]
        ) / 2
    else:
        median_z_score_population = z_score_list[total_pairs // 2]
    population_stat["population_median_z_score"] = median_z_score_population

    for pair, stats in statistics_dict.items():
        total_assignments = stats["total_assignments"]
        total_similarity_score = stats["total_similarity_score"]
        total_z_score = (
            total_similarity_score - total_assignments * mu
        ) / population_standard_deviation
        statistics_dict[pair]["total_z_score"] = total_z_score

    # for pair, scores in pairs_dict.items():

    return (statistics_dict, population_stat)


def populate_student_pair_stats(
    course_id: int,
    semester_id: int,
) -> tuple[dict[tuple[int, int], dict], dict]:
    """
    Run the full data‐analytics pipeline for a course & semester.

    Steps:
      1. Fetch all similarity pairs from the DB.
      2. Build an in‐memory dict of student‐pair → [(score, assignment), ...].
      3. Compute per‐pair and population statistics.
      4. Return (per_pair_stats, population_stats, duration_secs).

    Returns:
    per_pair_stats: {
           (student_id_1, student_id_2): {
            total_assignments: int,
            average_similarity_score: float,
            max_similarity_score: float,
            min_similarity_score: float,
            total_similarity_score: float,
            median_similarity_score: float,
            sample_standard_deviation: float,
            sample_variance: float,
            mean_z_score: float,
            max_z_score: float,
            min_z_score: float,
            total_z_score: float,
            flagged: bool
        },

      population_stats: {
          total_pairs: int,
          population_mean: float,
          population_variance: float,
          population_standard_deviation: float,
          population_median_similarity_score: float,
          population_median_z_score: float
      }
    """
    start_ts = time.time()
    logger.info(
        f"[PIPELINE] populate_student_pair_stats() for course={course_id}, semester={semester_id}"
    )

    # 1) Fetch raw pairs
    raw_pairs = get_all_similarity_pairs(course_id, semester_id)

    # 2) Build in-memory dict
    pairs_dict = turn_pairs_to_dict(raw_pairs)

    # 3) Compute stats
    per_pair_stats, population_stats = compute_pairs_statistics(pairs_dict)

    # 4) Measure duration
    duration = time.time() - start_ts
    logger.info(
        "[PIPELINE] populate_student_pair_stats() Completed in %.2f seconds", duration
    )

    # --- LOG FULL POPULATION STATS ---
    logger.info(
        "Population stats: total_pairs=%d | mean=%.4f | var=%.4f | std=%.4f | "
        "median_sim=%.4f | median_z=%.4f | mean_z=%.4f",
        population_stats["total_pairs"],
        population_stats["population_mean"],
        population_stats["population_variance"],
        population_stats["population_standard_deviation"],
        population_stats["population_median_similarity_score"],
        population_stats["population_median_z_score"],
        population_stats["population_mean_z_score"],
    )

    # populate here the model
    catalog = CourseCatalog.objects.get(pk=course_id)
    sem = Semester.objects.get(pk=semester_id)

    # 0) Pre-fetch previous flagged_count for each student pair
    flag_qs = PairFlagStat.objects.filter(
        course_catalog=catalog, semester=sem
    ).values_list("student_a_id", "student_b_id", "flagged_count")
    flagged_counts = {(a_id, b_id): count for a_id, b_id, count in flag_qs}

    to_create = []
    for (sid_a, sid_b), stats in per_pair_stats.items():
        # build an instance, but don’t save yet
        prev_flagged = flagged_counts.get((sid_a, sid_b), 0)
        inst = StudentPairSimilarityStatistics(
            course_catalog=catalog,
            semester=sem,
            student_a=Students.objects.get(pk=sid_a),
            student_b=Students.objects.get(pk=sid_b),
            assignments_shared=stats["total_assignments"],
            mean_similarity_score=stats["average_similarity_score"],
            median_similarity_score=stats["median_similarity_score"],
            similarity_std_dev=stats["sample_standard_deviation"],
            similarity_variance=stats["sample_variance"],
            min_similarity_score=stats["min_similarity_score"],
            max_similarity_score=stats["max_similarity_score"],
            mean_z_score=stats["mean_z_score"],
            min_z_score=stats["min_z_score"],
            max_z_score=stats["max_z_score"],
            total_similarity=stats["total_similarity_score"],
            total_z_score=stats["total_z_score"],
            pair_flagged=stats["flagged"],
            flagged_count=prev_flagged,
            # cluster_id, ta_notes, median_z_score, and flagged_count left blank for now
        )
        to_create.append(inst)

    # wipe out any old rows for this course+semester (optional)
    StudentPairSimilarityStatistics.objects.filter(
        course_catalog=catalog, semester=sem
    ).delete()
    PopulationSimilarityStatistics.objects.filter(
        course_catalog=catalog, semester=sem
    ).delete()

    ps = population_stats
    PopulationSimilarityStatistics.objects.create(
        course_catalog=catalog,
        semester=sem,
        total_unique_pairs=ps["total_pairs"],
        total_flagged_count=ps.get("total_flagged_count", 0),
        mean_similarity_score=ps["population_mean"],
        median_similarity_score=ps["population_median_similarity_score"],
        similarity_std_dev=ps["population_standard_deviation"],
        similarity_variance=ps["population_variance"],
        min_similarity_score=ps.get("population_min_similarity_score"),
        max_similarity_score=ps.get("population_max_similarity_score"),
        mean_z_score=ps.get("population_mean_z_score"),
        median_z_score=ps["population_median_z_score"],
        min_z_score=ps.get("population_min_z_score"),
        max_z_score=ps.get("population_max_z_score"),
    )

    # finally bulk‑insert
    StudentPairSimilarityStatistics.objects.bulk_create(to_create, batch_size=1000)

    return per_pair_stats, population_stats


def run_pair_level_clustering(
    course_id: int, semester_id: int, k_max: int = 6, B: int = 5, random_state: int = 0
) -> None:
    """
    1) Pulls all StudentPairSimilarityStatistics for the course+semester.

    2) Builds an N×D feature matrix (now including flagged_count, its square,
       and signed-square of max_z_score).
    3) Uses the Gap‐statistic to pick the number of clusters (≤ k_max).
    4) Fits KMeans, writes back `cluster_id` for each pair.
    """
    qs = StudentPairSimilarityStatistics.objects.filter(
        course_catalog_id=course_id, semester_id=semester_id
    ).order_by("id")

    if not qs.exists():
        return

    # build N×D array with our 9 features:
    X = np.vstack(
        [
            [
                obj.mean_similarity_score or 0.0,
                obj.median_similarity_score or 0.0,
                obj.similarity_std_dev or 0.0,
                obj.max_z_score or 0.0,
                obj.min_z_score or 0.0,
                obj.total_z_score or 0.0,
                obj.flagged_count or 0,
                (obj.flagged_count or 0) ** 2,
                (obj.max_z_score or 0.0) * abs(obj.max_z_score or 0.0),
            ]
            for obj in qs
        ]
    )

    # helper to get within‐cluster sum of squares
    def dispersion(data, k):
        km = KMeans(n_clusters=k, random_state=random_state).fit(data)
        return km.inertia_

    # compute gap‐statistic up to k_max
    def choose_k(data):
        n, d = data.shape
        mins, maxs = data.min(axis=0), data.max(axis=0)

        Wks = np.zeros(k_max)
        Wkbs = np.zeros((k_max, B))

        for k in range(1, k_max + 1):
            Wks[k - 1] = dispersion(data, k)
            for b in range(B):
                ref = np.random.uniform(mins, maxs, size=(n, d))
                Wkbs[k - 1, b] = dispersion(ref, k)

        logWks = np.log(Wks)
        logWkbs_mean = np.log(Wkbs).mean(axis=1)
        sk = np.sqrt(
            ((np.log(Wkbs) - logWkbs_mean[:, None]) ** 2).mean(axis=1)
        ) * np.sqrt(1 + 1 / B)
        gaps = logWkbs_mean - logWks

        for i in range(k_max - 1):
            if gaps[i] >= gaps[i + 1] - sk[i + 1]:
                return i + 1
        return k_max

    best_k = choose_k(X)

    # standardize features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    # fit & predict
    km = KMeans(n_clusters=best_k, random_state=random_state).fit(Xs)
    labels = km.labels_

    # write back in one bulk‐update
    with transaction.atomic():
        for obj, lbl in zip(qs, labels):
            obj.cluster_id = int(lbl)
        StudentPairSimilarityStatistics.objects.bulk_update(qs, ["cluster_id"])
