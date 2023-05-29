from operator import eq

"""排序算法"""


def bubble_sort(inp):
    for i in range(len(inp) - 1):
        for j in range(len(inp) - i - 1):
            if inp[j] > inp[j + 1]:
                inp[j], inp[j + 1] = inp[j + 1], inp[j]
    print(inp)


def select_sort(inp):
    for i in range(len(inp) - 1):
        max_idx = 0
        for j in range(1, len(inp) - i):
            if inp[j] > inp[max_idx]:
                max_idx = j
        inp[max_idx], inp[len(inp) - i - 1] = inp[len(inp) - i - 1], inp[max_idx]
    return inp


def insert_sort(inp):
    for i in range(1, len(inp)):
        temp = inp[i]
        for j in range(0, i)[::-1]:
            if inp[j] > temp:
                inp[j + 1] = inp[j]
            else:
                break
        inp[j] = temp
    return inp


def merge_sort(inp):
    merge_sort_r(inp, 0, len(inp))


def merge_sort_r(inp, start, end):
    if start + 1 >= end:
        return
    else:
        mid = (start + end) // 2
        merge_sort_r(inp, start, mid)
        merge_sort_r(inp, mid, end)
        merge(inp, start, mid, end)


def merge(inp, start, mid, end):
    tmp = []
    i = start
    j = mid
    while i < mid and j < end:
        if inp[i] <= inp[j]:
            tmp.append(inp[i])
            i += 1
        else:
            tmp.append(inp[j])
            j += 1
    while i < mid:
        tmp.append(inp[i])
        i += 1
    while j < end:
        tmp.append(inp[j])
        j += 1
    inp[start:end] = tmp[:]


def quick_sort(inp, l, r):
    if l >= r:
        return
    low = l
    high = r
    pivot = inp[l]
    while l < r:
        while l < r and inp[r] > pivot:
            r -= 1
        inp[l] = inp[r]
        while l < r and inp[l] <= pivot:
            l += 1
        inp[r] = inp[l]

    inp[r] = pivot
    quick_sort(inp, low + 1, r - 1)
    quick_sort(inp, r + 1, high)


# inp = [1, 3, 2, 6, 5, 8, 24, 9, 34]
# answer = [1, 2, 3, 5, 6, 8, 9, 24, 34]
# # inp_new = merge_sort(inp)
# # print(eq(answer, inp_new))
# quick_sort(inp, 0, len(inp) - 1)
# print(inp)


def resort(probs):
    """对概率重排，返回排名结果"""
    new_ind = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)
    new_ind = sorted(enumerate(new_ind), key=lambda x: x[1])
    return [i for i, _ in new_ind]

