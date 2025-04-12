import time

def classical_search(dataset, target, method):
    start = time.time()
    found = False
    if method == "linear":
        found = target in dataset
    elif method == "binary":
        dataset.sort()
        left, right = 0, len(dataset) - 1
        while left <= right:
            mid = (left + right) // 2
            if dataset[mid] == target:
                found = True
                break
            elif dataset[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
    end = time.time()
    return {
        "method": method,
        "target": target,
        "found": found,
        "time": round(end - start, 6)
    }
