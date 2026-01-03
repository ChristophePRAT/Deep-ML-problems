def distance(point1: list[float], point2: list[float]) -> float:
    return sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))])


def get_regions(points: list[list[float]], centroids: list[list[float]]) -> list[int]:
    regions = []
    for point in points:
        distances = [distance(point, centroid) for centroid in centroids]
        region = distances.index(min(distances))
        regions.append(region)
    return regions


def mean_point(points: list[list[float]]) -> list[float]:
    if not points:
        return [0 for _ in points[0]]
    # x_mean = sum(p[0] for p in points) / len(points)
    # y_mean = sum(p[1] for p in points) / len(points)
    # return (x_mean, y_mean)

    dim = len(points[0])
    mean = []
    for i in range(dim):
        coord_mean = sum(p[i] for p in points) / len(points)
        mean.append(coord_mean)
    return mean


def k_means_clustering(
    points: list[list[float]],
    k: int,
    initial_centroids: list[list[float]],
    max_iterations: int,
) -> list[list[float]]:
    centroids = initial_centroids.copy()

    for _ in range(max_iterations):
        regions = get_regions(points, centroids)

        new_centroids = []

        for i in range(k):
            cluster_points = [points[j] for j in range(len(points)) if regions[j] == i]
            new_centroid = mean_point(cluster_points)
            new_centroids.append(new_centroid)
        centroids = new_centroids
    return centroids
