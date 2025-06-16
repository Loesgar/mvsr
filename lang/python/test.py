import mvsr

print(mvsr.__file__)

x1 = range(1, 21)
y1 = [1, 2, 3, 4, 5, 6, 7, 8, 2, 2, 2, 2, 2, 2, 1, 0, -1, -2, -3, -4]
y2 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

mvsr.segreg(x1, y1, 3)
mvsr.segreg(x1, y1, 3, normalize=True)
print(mvsr.segreg(x1, [y1, y2], 3)[0].get_samplecount())
print(mvsr.segreg(x1, [y1, y2], 3, weighting=[0.1, 1000000.0])[0].get_samplecount())
print(mvsr.segreg(x1, y2, 3)[0].get_samplecount())
