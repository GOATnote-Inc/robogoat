// action_space_conversion.cu
// GPU-accelerated action space conversion
// Phase 4 implementation - Forward/Inverse Kinematics and Jacobian

#include "action_space_conversion.h"
#include <cuda_bf16.h>
#include <cmath>

namespace robocache {
namespace kernels {

constexpr int BLOCK_SIZE = 256;

//==============================================================================
// Matrix utilities
//==============================================================================

struct Mat4 {
    float m[16]; // Row-major 4x4 matrix
    
    __device__ __forceinline__
    Mat4() {
        for (int i = 0; i < 16; i++) m[i] = 0.0f;
        m[0] = m[5] = m[10] = m[15] = 1.0f; // Identity
    }
    
    __device__ __forceinline__
    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i*4+j] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    result.m[i*4+j] += m[i*4+k] * other.m[k*4+j];
                }
            }
        }
        return result;
    }
};

// DH transformation matrix
__device__ __forceinline__
Mat4 dh_transform(float a, float alpha, float d, float theta) {
    Mat4 T;
    float ct = cosf(theta);
    float st = sinf(theta);
    float ca = cosf(alpha);
    float sa = sinf(alpha);
    
    T.m[0]  = ct;  T.m[1]  = -st*ca; T.m[2]  = st*sa;  T.m[3]  = a*ct;
    T.m[4]  = st;  T.m[5]  = ct*ca;  T.m[6]  = -ct*sa; T.m[7]  = a*st;
    T.m[8]  = 0;   T.m[9]  = sa;     T.m[10] = ca;     T.m[11] = d;
    T.m[12] = 0;   T.m[13] = 0;      T.m[14] = 0;      T.m[15] = 1;
    
    return T;
}

// Convert rotation matrix to quaternion
__device__ __forceinline__
void mat_to_quat(const Mat4& T, float& qx, float& qy, float& qz, float& qw) {
    float trace = T.m[0] + T.m[5] + T.m[10];
    
    if (trace > 0) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        qw = 0.25f * s;
        qx = (T.m[9] - T.m[6]) / s;
        qy = (T.m[2] - T.m[8]) / s;
        qz = (T.m[4] - T.m[1]) / s;
    } else if ((T.m[0] > T.m[5]) && (T.m[0] > T.m[10])) {
        float s = sqrtf(1.0f + T.m[0] - T.m[5] - T.m[10]) * 2.0f;
        qw = (T.m[9] - T.m[6]) / s;
        qx = 0.25f * s;
        qy = (T.m[1] + T.m[4]) / s;
        qz = (T.m[2] + T.m[8]) / s;
    } else if (T.m[5] > T.m[10]) {
        float s = sqrtf(1.0f + T.m[5] - T.m[0] - T.m[10]) * 2.0f;
        qw = (T.m[2] - T.m[8]) / s;
        qx = (T.m[1] + T.m[4]) / s;
        qy = 0.25f * s;
        qz = (T.m[6] + T.m[9]) / s;
    } else {
        float s = sqrtf(1.0f + T.m[10] - T.m[0] - T.m[5]) * 2.0f;
        qw = (T.m[4] - T.m[1]) / s;
        qx = (T.m[2] + T.m[8]) / s;
        qy = (T.m[6] + T.m[9]) / s;
        qz = 0.25f * s;
    }
}

//==============================================================================
// Forward Kinematics Kernel
//==============================================================================

__global__ void forward_kinematics_kernel(
    const float* __restrict__ joint_angles,
    float* __restrict__ ee_poses,
    const float* __restrict__ dh_params,
    int batch_size,
    int num_joints
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Compute forward kinematics for this batch item
    Mat4 T_total; // Identity
    
    for (int j = 0; j < num_joints; j++) {
        // Load DH parameters: a, alpha, d, theta_offset
        float a = dh_params[j * 4 + 0];
        float alpha = dh_params[j * 4 + 1];
        float d = dh_params[j * 4 + 2];
        float theta_offset = dh_params[j * 4 + 3];
        
        // Get joint angle
        float theta = joint_angles[batch_idx * num_joints + j] + theta_offset;
        
        // Compute DH transformation
        Mat4 T_j = dh_transform(a, alpha, d, theta);
        
        // Accumulate transformation
        T_total = T_total * T_j;
    }
    
    // Extract position (x, y, z)
    float x = T_total.m[3];
    float y = T_total.m[7];
    float z = T_total.m[11];
    
    // Extract orientation as quaternion (qx, qy, qz, qw)
    float qx, qy, qz, qw;
    mat_to_quat(T_total, qx, qy, qz, qw);
    
    // Write output: [x, y, z, qx, qy, qz, qw]
    int out_offset = batch_idx * 7;
    ee_poses[out_offset + 0] = x;
    ee_poses[out_offset + 1] = y;
    ee_poses[out_offset + 2] = z;
    ee_poses[out_offset + 3] = qx;
    ee_poses[out_offset + 4] = qy;
    ee_poses[out_offset + 5] = qz;
    ee_poses[out_offset + 6] = qw;
}

//==============================================================================
// Jacobian Computation Kernel (Numerical differentiation)
//==============================================================================

__global__ void compute_jacobian_kernel(
    const float* __restrict__ joint_angles,
    float* __restrict__ jacobians,
    const float* __restrict__ dh_params,
    int batch_size,
    int num_joints
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    constexpr float epsilon = 1e-5f;
    
    // Compute FK for current configuration
    Mat4 T_center;
    for (int j = 0; j < num_joints; j++) {
        float a = dh_params[j * 4 + 0];
        float alpha = dh_params[j * 4 + 1];
        float d = dh_params[j * 4 + 2];
        float theta_offset = dh_params[j * 4 + 3];
        float theta = joint_angles[batch_idx * num_joints + j] + theta_offset;
        Mat4 T_j = dh_transform(a, alpha, d, theta);
        T_center = T_center * T_j;
    }
    
    float x_center = T_center.m[3];
    float y_center = T_center.m[7];
    float z_center = T_center.m[11];
    
    // Compute numerical Jacobian column-by-column
    for (int j = 0; j < num_joints; j++) {
        // Perturb joint j by epsilon
        Mat4 T_perturb;
        for (int k = 0; k < num_joints; k++) {
            float a = dh_params[k * 4 + 0];
            float alpha = dh_params[k * 4 + 1];
            float d = dh_params[k * 4 + 2];
            float theta_offset = dh_params[k * 4 + 3];
            float theta = joint_angles[batch_idx * num_joints + k] + theta_offset;
            if (k == j) theta += epsilon; // Perturb this joint
            Mat4 T_k = dh_transform(a, alpha, d, theta);
            T_perturb = T_perturb * T_k;
        }
        
        float x_perturb = T_perturb.m[3];
        float y_perturb = T_perturb.m[7];
        float z_perturb = T_perturb.m[11];
        
        // Compute finite difference for linear velocity
        float dx_dq = (x_perturb - x_center) / epsilon;
        float dy_dq = (y_perturb - y_center) / epsilon;
        float dz_dq = (z_perturb - z_center) / epsilon;
        
        // Store in Jacobian: [batch, 6, num_joints]
        // For simplicity, only computing linear velocity part (first 3 rows)
        // Angular velocity would require rotation differentiation
        int jac_offset = batch_idx * (6 * num_joints);
        jacobians[jac_offset + 0 * num_joints + j] = dx_dq;
        jacobians[jac_offset + 1 * num_joints + j] = dy_dq;
        jacobians[jac_offset + 2 * num_joints + j] = dz_dq;
        
        // Angular velocity part (simplified - set to zero for now)
        jacobians[jac_offset + 3 * num_joints + j] = 0.0f;
        jacobians[jac_offset + 4 * num_joints + j] = 0.0f;
        jacobians[jac_offset + 5 * num_joints + j] = 0.0f;
    }
}

//==============================================================================
// Host API
//==============================================================================

cudaError_t forward_kinematics(
    const float* joint_angles,
    float* ee_poses,
    const float* dh_params,
    int batch_size,
    int num_joints,
    cudaStream_t stream
) {
    int num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    forward_kinematics_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        joint_angles, ee_poses, dh_params, batch_size, num_joints
    );
    return cudaGetLastError();
}

cudaError_t compute_jacobian(
    const float* joint_angles,
    float* jacobians,
    const float* dh_params,
    int batch_size,
    int num_joints,
    cudaStream_t stream
) {
    int num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_jacobian_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        joint_angles, jacobians, dh_params, batch_size, num_joints
    );
    return cudaGetLastError();
}

// Stub implementations for IK and other functions (to be completed)
cudaError_t inverse_kinematics(
    const float* target_poses,
    const float* joint_angles_init,
    float* joint_angles_out,
    const float* dh_params,
    int batch_size,
    int num_joints,
    int max_iterations,
    float tolerance,
    float damping,
    cudaStream_t stream
) {
    // TODO: Implement iterative IK solver (Levenberg-Marquardt)
    // For now, return not implemented
    return cudaErrorNotSupported;
}

cudaError_t cartesian_to_joint_velocity(
    const float* cartesian_velocities,
    const float* joint_angles,
    float* joint_velocities,
    const float* dh_params,
    int batch_size,
    int num_joints,
    cudaStream_t stream
) {
    // TODO: Implement using Jacobian pseudoinverse
    return cudaErrorNotSupported;
}

cudaError_t convert_action_space(
    const float* actions,
    float* converted_actions,
    const float* joint_angles_current,
    const float* dh_params,
    int batch_size,
    int num_joints,
    bool joint_to_cartesian,
    cudaStream_t stream
) {
    // TODO: Implement action space conversion
    return cudaErrorNotSupported;
}

} // namespace kernels
} // namespace robocache

