from sympy import *
# from sympy import symbols

def binom(n, i):
	if n == 0 or i == 0 or i == n: 
		return 1
	res = 1
	while (i >= 0):
		res *= n - i
		i -= 1
	return res

def complete_bell_polynomial(n, xs):
	if n == 0:
		return 1

	res = 0
	for i in range(n):
		c = binom(n - 1, i)
		# print(i, n, xs)
		# print(i + 1)
		val = complete_bell_polynomial(n - i - 1, xs[:n - i - 1])
		# print(i, n, c, val, xs[i])
		res += c * val * xs[i]
	return res

def prediction(alpha, n):
	if n == 1:
		return [1]
	Q_prev = prediction(alpha, n-1)
	Q[1] = Q_prev[1] + n * (n + 1)
	Q = alpha

def main():
	alpha = symbols(r'\alpha')
	# init_printing(use_unicode=True)

	n = 10


	for i in range(1, n):
		xs = [j - alpha for j in range(i)]
		xs[0] = alpha
		B = complete_bell_polynomial(i, xs)
		# B = collect(B, alpha)
		B = expand(B)
		print(i, B)

if __name__ == '__main__':
	main()