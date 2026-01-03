import torch


def svd_2x2_singular_values(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 torch tensor

    Returns:
        Tuple (U, S, Vt) where A â U @ diag(S) @ Vt
    """

    # Get singular values and compute S

    at_a = A.T @ A
    trace = at_a[0, 0] + at_a[1, 1]
    determinant = at_a[0, 0] * at_a[1, 1] - at_a[0, 1] * at_a[1, 0]

    discriminant = trace**2 - 4 * determinant
    sqrt_discriminant = torch.sqrt(discriminant)

    sigma1 = torch.sqrt((trace + sqrt_discriminant) / 2)
    sigma2 = torch.sqrt((trace - sqrt_discriminant) / 2)

    S = torch.stack([sigma1, sigma2])

    # Compute Vt

    v1 = torch.tensor([at_a[0, 1], sigma1**2 - at_a[0, 0]])
    v2 = torch.tensor([at_a[0, 1], sigma2**2 - at_a[0, 0]])

    if torch.norm(v1) > 0:
        v1 = v1 / torch.norm(v1)
    else:
        v1 = torch.tensor([1.0, 0.0])

    if torch.norm(v2) > 0:
        v2 = v2 / torch.norm(v2)
    else:
        v2 = torch.tensor([0.0, 1.0])

    V = torch.stack([v1, v2], dim=1)
    Vt = V.T

    # Compute U
    if sigma1 > 0:
        u1 = (1 / sigma1) * A @ V[:, 0]
    else:
        u1 = torch.tensor([1.0, 0.0])

    if sigma2 > 0:
        u2 = (1 / sigma2) * A @ V[:, 1]
    else:
        # complete the orthonormal basis
        u2 = torch.tensor([-u1[1], u1[0]])

    if torch.norm(u1) > 0:
        u1 = u1 / torch.norm(u1)
    if torch.norm(u2) > 0:
        u2 = u2 / torch.norm(u2)

    U = torch.stack([u1, u2], dim=1)
    return U, S, Vt


# Tests
def test():
    # Test 1: Simple diagonal matrix
    A1 = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    U1, S1, Vt1 = svd_2x2_singular_values(A1)

    # Verify reconstruction
    reconstructed1 = U1 @ torch.diag(S1) @ Vt1
    assert torch.allclose(A1, reconstructed1, atol=1e-5), (
        f"Test 1 reconstruction failed: {A1} vs {reconstructed1}"
    )

    # Compare with PyTorch SVD
    U1_ref, S1_ref, Vt1_ref = torch.linalg.svd(A1)
    assert torch.allclose(torch.sort(S1)[0], torch.sort(S1_ref)[0], atol=1e-5), (
        f"Test 1 singular values mismatch: {S1} vs {S1_ref}"
    )
    print("Test 1 passed: Diagonal matrix")

    # Test 2: General matrix
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    U2, S2, Vt2 = svd_2x2_singular_values(A2)

    # Verify reconstruction
    reconstructed2 = U2 @ torch.diag(S2) @ Vt2
    assert torch.allclose(A2, reconstructed2, atol=1e-5), (
        f"Test 2 reconstruction failed: {A2} vs {reconstructed2}"
    )

    # Compare with PyTorch SVD
    U2_ref, S2_ref, Vt2_ref = torch.linalg.svd(A2)
    assert torch.allclose(torch.sort(S2)[0], torch.sort(S2_ref)[0], atol=1e-5), (
        f"Test 2 singular values mismatch: {S2} vs {S2_ref}"
    )
    print("Test 2 passed: General matrix")

    # Test 3: Symmetric matrix
    A3 = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
    U3, S3, Vt3 = svd_2x2_singular_values(A3)

    # Verify reconstruction
    reconstructed3 = U3 @ torch.diag(S3) @ Vt3
    assert torch.allclose(A3, reconstructed3, atol=1e-5), (
        f"Test 3 reconstruction failed: {A3} vs {reconstructed3}"
    )

    # Compare with PyTorch SVD
    U3_ref, S3_ref, Vt3_ref = torch.linalg.svd(A3)
    assert torch.allclose(torch.sort(S3)[0], torch.sort(S3_ref)[0], atol=1e-5), (
        f"Test 3 singular values mismatch: {S3} vs {S3_ref}"
    )
    print("Test 3 passed: Symmetric matrix")

    # Test 4: Matrix with negative values
    A4 = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    U4, S4, Vt4 = svd_2x2_singular_values(A4)

    # Verify reconstruction
    reconstructed4 = U4 @ torch.diag(S4) @ Vt4
    assert torch.allclose(A4, reconstructed4, atol=1e-5), (
        f"Test 4 reconstruction failed: {A4} vs {reconstructed4}"
    )

    # Compare with PyTorch SVD
    U4_ref, S4_ref, Vt4_ref = torch.linalg.svd(A4)
    assert torch.allclose(torch.sort(S4)[0], torch.sort(S4_ref)[0], atol=1e-5), (
        f"Test 4 singular values mismatch: {S4} vs {S4_ref}"
    )
    print("Test 4 passed: Matrix with negative values")

    # Test 5: Random matrix
    torch.manual_seed(42)
    A5 = torch.randn(2, 2)
    U5, S5, Vt5 = svd_2x2_singular_values(A5)

    # Verify reconstruction
    reconstructed5 = U5 @ torch.diag(S5) @ Vt5
    assert torch.allclose(A5, reconstructed5, atol=1e-5), (
        f"Test 5 reconstruction failed: {A5} vs {reconstructed5}"
    )

    # Compare with PyTorch SVD
    U5_ref, S5_ref, Vt5_ref = torch.linalg.svd(A5)
    assert torch.allclose(torch.sort(S5)[0], torch.sort(S5_ref)[0], atol=1e-5), (
        f"Test 5 singular values mismatch: {S5} vs {S5_ref}"
    )
    print("Test 5 passed: Random matrix")

    # Test 6: Verify U and V are orthonormal
    assert torch.allclose(U5.T @ U5, torch.eye(2), atol=1e-5), "U is not orthonormal"
    assert torch.allclose(Vt5 @ Vt5.T, torch.eye(2), atol=1e-5), "V is not orthonormal"
    print("Test 6 passed: Orthonormality check")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test()
