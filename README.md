# MEM704 - Machine Learning
## 1st Laboratory Exercise
### The PageRank Algorithm and SVD Analysis
**Submission Deadline:** 07/03/2024, 18:00  
**Examination:** 08/03/2024 in the Lab  

In this exercise, we will implement the PageRank algorithm to estimate the importance of web pages on the internet. At the core of this algorithm is the power method for finding the dominant eigenvalue of an appropriate matrix. We will also use Singular Value Decomposition (SVD) for image reconstruction.

---
## 1. Power Method
Let \( A \in \mathbb{R}^{n \times n} \). The power method for approximating the dominant eigenvalue \( \lambda \) and its corresponding eigenvector \( x \) is given by the following pseudocode:

### **Power Method for Ax = λx**
1. Choose a random \( x_0 \in \mathbb{R}^n \), normalize it: \( x_0 = \frac{x_0}{\|x_0\|} \)
2. Set \( k_{max} \) (maximum iterations)
3. Set \( \epsilon \approx 10^{-6} \) (error threshold)
4. Initialize \( k = 0, d_k = 1 \)
5. **While** \( d_k > \epsilon \) and \( k < k_{max} \):
   - \( x_k = A x_{k-1} \)
   - Normalize: \( x_k = \frac{x_k}{\|x_k\|} \)
   - Compute difference: \( d_k = \|x_k - x_{k-1}\| \)
   - Compute eigenvalue: \( \lambda = \frac{x_k^T A x_k}{x_k^T x_k} \)

**Task:** Implement this algorithm in Python using NumPy's `linalg` package. For initial vector \( x_0 \), use NumPy's `random` package. Test your implementation on an example matrix (\( n \geq 3 \)) with known eigenvalues and eigenvectors.

Verify the theoretical convergence speed by comparing the ratio \( |\lambda_2 / \lambda_1| \) with successive ratios \( d_{k+1} / d_k \).

Apply your implementation to the tridiagonal matrix:
\[
T = \begin{bmatrix}
2 & -1 & 0 & \dots & 0 \\
-1 & 2 & -1 & \dots & 0 \\
0 & -1 & 2 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & -1 \\
0 & 0 & 0 & -1 & 2
\end{bmatrix}
\]
for various values of \( n \). The eigenvalues of \( T \) are given by:
\[
\lambda_k = 2 - 2 \cos\left( \frac{k \pi}{n+1} \right), \quad k = 1, ..., n.
\]

---
## 2. PageRank Algorithm
The PageRank algorithm is the core of Google's search engine and estimates the importance of a web page. It was proposed by S. Brin and L. Page in 1998 in "The Anatomy of a Large-Scale Hypertextual Web Search Engine."

The method relies on finding the dominant eigenvector of a Markov matrix. A Markov matrix is square, its elements represent probabilities, and each column sums to 1. These properties ensure that 1 is the largest eigenvalue with multiplicity 1.

Given \( N \) web pages and a link structure, the Markov matrix is:
\[
M = d \cdot A + \frac{(1-d)}{N} B,
\]
where:
- \( d \approx 0.85 \) (damping factor)
- \( N \) is the number of web pages
- \( A \) is defined as:
  \[
  a_{ij} = \begin{cases}
  \frac{1}{L(j)}, & \text{if there is a link from page } j \text{ to page } i \\
  0, & \text{otherwise}
  \end{cases}
  \]
- \( L(j) \) is the number of outbound links from page \( j \).

### **Task:**
1. Implement a Python program that reads a graph structure from a file and constructs matrix \( M \).
2. Use the power method to compute the dominant eigenvector and eigenvalue.
3. Sort and print web pages in descending order of importance.

Test your implementation using:
```txt
graph0.txt:
1 2
1 3
1 4
2 3
2 4
3 1
4 1
4 3
```
and compare results with `networkx.pagerank(G, d)`.

---
## 3. SVD Analysis & Image Reconstruction
Given an image, we will use Singular Value Decomposition (SVD) to reconstruct it using minimal data.

Using the decomposition:
\[
A = U \Sigma V^T
\]
We approximate \( A \) using \( k \) singular values:
\[
A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T
\]
where \( A_k \) is the best rank-\( k \) approximation of \( A \).

### **Task:**
1. Implement a Python program to:
   - Read an image \( A \)
   - Compute its SVD
   - Construct \( A_k \) for various values of \( k \)
2. Compute error: \( \epsilon_k = \| A - A_k \|_2 \)
3. Plot:
   - The sequence \( (k, \epsilon_k) \)
   - Singular values \( \sigma_i \)
   - The original image and approximations \( A_k \) for selected \( k \).

Use `matplotlib.image.imread` for image reading and `numpy.linalg.svd` for SVD computation.

Test with:
- `uoc_logo.png`
- `python_logo.png`

---
## 4. Submission & Examination
- Create separate Python scripts for each task, named:
  ```
  {math, tem, ph}XXXX_Lab1{a, b, c, d}.py
  ```
  where `XXXX` is your student ID.
- Include your name, surname, and ID as comments at the top of each file.
- Submit as `{math, tem, ph}XXXX_LAB1.zip` on **UoC-eLearn** by **07 March 2024, 18:00**.
- Late submissions **will not be graded**.
- **Individual work required**. Plagiarized code **will receive a zero**.

