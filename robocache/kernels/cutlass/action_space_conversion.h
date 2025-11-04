// action_space_conversion.h
// GPU-accelerated action space conversion for robot manipulation
// Phase 4: Forward/Inverse Kinematics and Jacobian computation

#ifndef ROBOCACHE_ACTION_SPACE_CONVERSION_H
#define ROBOCACHE_ACTION_SPACE_CONVERSION_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace robocache {
namespace kernels {

/**
 * Action Space Conversion for Robot Manipulation
 * 
 * Critical operations for robot control:
 * - Forward Kinematics (FK): Joint angles → End-effector pose
 * - Inverse Kinematics (IK): End-effector pose → Joint angles  
 * - Jacobian: Maps joint velocities to Cartesian velocities
 * 
 * Use cases:
 * - Policy learning (map actions to robot commands)
 * - Trajectory optimization (Cartesian ↔ Joint space)
 * - Force control (Jacobian transpose for impedance)
 * 
 * Performance target: 50-100x faster than CPU
 */

/**
 * Forward Kinematics (FK) - Batch
 * 
 * Compute end-effector pose from joint angles using DH parameters.
 * 
 * Args:
 *   joint_angles: [batch, num_joints] - Input joint configuration
 *   ee_poses: [batch, 7] - Output end-effector poses (x, y, z, qx, qy, qz, qw)
 *   dh_params: [num_joints, 4] - DH parameters (a, alpha, d, theta_offset)
 *   batch_size: Number of configurations
 *   num_joints: Number of robot joints (typically 6-7 for manipulators)
 *   stream: CUDA stream
 * 
 * Returns: cudaSuccess or error code
 */
cudaError_t forward_kinematics(
    const float* joint_angles,
    float* ee_poses,
    const float* dh_params,
    int batch_size,
    int num_joints,
    cudaStream_t stream = 0
);

/**
 * Compute Jacobian Matrix - Batch
 * 
 * Compute geometric Jacobian J(q) that maps joint velocities to Cartesian velocities:
 * v = J(q) * dq/dt
 * 
 * Args:
 *   joint_angles: [batch, num_joints] - Current joint configuration
 *   jacobians: [batch, 6, num_joints] - Output Jacobian matrices
 *   dh_params: [num_joints, 4] - DH parameters
 *   batch_size: Number of configurations
 *   num_joints: Number of robot joints
 *   stream: CUDA stream
 * 
 * Returns: cudaSuccess or error code
 * 
 * Note: Jacobian is 6xN (3 linear velocity + 3 angular velocity) × N joints
 */
cudaError_t compute_jacobian(
    const float* joint_angles,
    float* jacobians,
    const float* dh_params,
    int batch_size,
    int num_joints,
    cudaStream_t stream = 0
);

/**
 * Inverse Kinematics (IK) - Batch Numerical Solver
 * 
 * Solve for joint angles that achieve desired end-effector pose.
 * Uses damped least-squares (Levenberg-Marquardt) on GPU.
 * 
 * Args:
 *   target_poses: [batch, 7] - Desired end-effector poses (x, y, z, qx, qy, qz, qw)
 *   joint_angles_init: [batch, num_joints] - Initial guess for IK
 *   joint_angles_out: [batch, num_joints] - Solved joint angles (output)
 *   dh_params: [num_joints, 4] - DH parameters
 *   batch_size: Number of IK problems to solve
 *   num_joints: Number of robot joints
 *   max_iterations: Maximum iterations per IK solve (default: 100)
 *   tolerance: Convergence tolerance in meters (default: 1e-4)
 *   damping: Damping factor for damped least squares (default: 0.01)
 *   stream: CUDA stream
 * 
 * Returns: cudaSuccess or error code
 * 
 * Note: IK is iterative and may not converge for all poses.
 *       Check residual error to verify convergence.
 */
cudaError_t inverse_kinematics(
    const float* target_poses,
    const float* joint_angles_init,
    float* joint_angles_out,
    const float* dh_params,
    int batch_size,
    int num_joints,
    int max_iterations = 100,
    float tolerance = 1e-4f,
    float damping = 0.01f,
    cudaStream_t stream = 0
);

/**
 * Cartesian to Joint Space Velocity
 * 
 * Map Cartesian velocity to joint velocity using Jacobian pseudoinverse:
 * dq/dt = J^+ * v
 * 
 * Args:
 *   cartesian_velocities: [batch, 6] - Desired Cartesian velocities
 *   joint_angles: [batch, num_joints] - Current joint configuration
 *   joint_velocities: [batch, num_joints] - Output joint velocities
 *   dh_params: [num_joints, 4] - DH parameters
 *   batch_size: Number of velocity mappings
 *   num_joints: Number of robot joints
 *   stream: CUDA stream
 * 
 * Returns: cudaSuccess or error code
 */
cudaError_t cartesian_to_joint_velocity(
    const float* cartesian_velocities,
    const float* joint_angles,
    float* joint_velocities,
    const float* dh_params,
    int batch_size,
    int num_joints,
    cudaStream_t stream = 0
);

/**
 * Joint Space to Cartesian Space - Convert action
 * 
 * Utility to convert actions from one space to another.
 * Useful for policy learning when action space doesn't match control space.
 * 
 * Args:
 *   actions: [batch, action_dim] - Input actions (joint or Cartesian)
 *   converted_actions: [batch, converted_dim] - Output actions
 *   joint_angles_current: [batch, num_joints] - Current robot configuration
 *   dh_params: [num_joints, 4] - DH parameters
 *   batch_size: Number of actions to convert
 *   num_joints: Number of robot joints
 *   joint_to_cartesian: true = Joint→Cartesian, false = Cartesian→Joint
 *   stream: CUDA stream
 * 
 * Returns: cudaSuccess or error code
 */
cudaError_t convert_action_space(
    const float* actions,
    float* converted_actions,
    const float* joint_angles_current,
    const float* dh_params,
    int batch_size,
    int num_joints,
    bool joint_to_cartesian,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace robocache

#endif // ROBOCACHE_ACTION_SPACE_CONVERSION_H

