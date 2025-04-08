import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D

# label2key dictionary
label2key = {0:"Nose",
             1:"Left Eye",
             2:"Right Eye",
             3:"Left Ear",
             4:"Right Ear",
             5:"Left Shoulder",
             6:"Right Shoulder",
             7:"Left Elbow",
             8:"Right Elbow",
             9:"Left Wrist",
             10:"Right Wrist",
             11:"Left Hip",
             12:"Right Hip",
             13:"Left Knee",
             14:"Right Knee",
             15:"Left Ankle",
             16:"Right Ankle"}

# ----- 데이터 로드 함수 -----
def load_2d_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("관절 키들:", list(data['data'].keys()))
    skeleton = []
    for key in sorted(data['data'].keys(), key=lambda x: int(x)):
        pts = np.array(data['data'][key])[:, :2]  # (n_frames, 2)
        skeleton.append(pts)
    skeleton = np.stack(skeleton, axis=0)  # (num_joints, n_frames, 2)
    
    conf = []
    for key in sorted(data['conf'].keys(), key=lambda x: int(x)):
        c = np.array(data['conf'][key])  # (n_frames,)
        conf.append(c)
    conf = np.stack(conf, axis=0)  # (num_joints, n_frames)
    return skeleton, conf

# ----- 초기 3D 추정 (단순 모델) -----
def initial_reconstruct_3d(skel_front, skel_left, skel_right):
    num_joints, n_frames, _ = skel_front.shape
    recon = np.zeros((num_joints, n_frames, 3))
    for j in range(num_joints):
        for i in range(n_frames):
            X = skel_front[j, i, 0]
            Y = (skel_front[j, i, 1] + skel_left[j, i, 1] + skel_right[j, i, 1]) / 3.0
            Z = (skel_left[j, i, 0] - skel_right[j, i, 0]) / 2.0
            recon[j, i, :] = np.array([X, Y, Z])
    return recon

# ----- DLT를 통한 카메라 프로젝션 행렬 추정 -----
def aggregate_correspondences(initial_3d, skeleton_2d, conf, threshold):
    num_joints, n_frames, _ = skeleton_2d.shape
    X_list, p_list = [], []
    for j in range(num_joints):
        for i in range(n_frames):
            if conf[j, i] >= threshold:
                X = initial_3d[j, i, :]
                X_h = np.hstack([X, 1])  # (4,)
                p = skeleton_2d[j, i, :]  # (2,)
                X_list.append(X_h)
                p_list.append(p)
    return np.array(X_list), np.array(p_list)

def estimate_projection_matrix(X, p):
    N = X.shape[0]
    A = []
    b = []
    for i in range(N):
        X_i = X[i, :]
        u, v = p[i, 0], p[i, 1]
        A.append(X_i)
        b.append(u)
        A.append(X_i)
        b.append(v)
    A = np.array(A)
    b = np.array(b)
    A_u = A[0::2]
    b_u = b[0::2]
    P0, _, _, _ = lstsq(A_u, b_u)
    A_v = A[1::2]
    b_v = b[1::2]
    P1, _, _, _ = lstsq(A_v, b_v)
    P = np.vstack([P0, P1])
    return P

# ----- 수정된 삼각측량 함수 (u, v 모두 활용) -----
def triangulate_point(p_list, P_list):
    A = []
    for p, P in zip(p_list, P_list):
        # 2x4 행렬 P를 3x4로 확장: 3번째 행을 [0,0,0,1]로 추가
        Q = np.vstack([P, np.array([0, 0, 0, 1])])
        u, v = p[0], p[1]
        A.append(u * Q[2, :] - Q[0, :])
        A.append(v * Q[2, :] - Q[1, :])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]
    X_h = X_h / X_h[-1]
    return X_h[:3]

# ----- 메인 처리 -----
front_path = './output/1-frontcut/1-frontcut.pkl'
left_path  = './output/1-leftcut/1-leftcut.pkl'
right_path = './output/1-rightcut/1-rightcut.pkl'

skel_front, conf_front = load_2d_data(front_path)
skel_left, conf_left   = load_2d_data(left_path)
skel_right, conf_right = load_2d_data(right_path)

n_frames_common = min(skel_front.shape[1], skel_left.shape[1], skel_right.shape[1])
skel_front = skel_front[:, :n_frames_common, :]
skel_left  = skel_left[:, :n_frames_common, :]
skel_right = skel_right[:, :n_frames_common, :]
conf_front = conf_front[:, :n_frames_common]
conf_left  = conf_left[:, :n_frames_common]
conf_right = conf_right[:, :n_frames_common]

initial_3d = initial_reconstruct_3d(skel_front, skel_left, skel_right)

threshold = 0.8
X_front, p_front = aggregate_correspondences(initial_3d, skel_front, conf_front, threshold)
X_left,  p_left  = aggregate_correspondences(initial_3d, skel_left, conf_left, threshold)
X_right, p_right = aggregate_correspondences(initial_3d, skel_right, conf_right, threshold)

P_front = estimate_projection_matrix(X_front, p_front)
P_left  = estimate_projection_matrix(X_left, p_left)
P_right = estimate_projection_matrix(X_right, p_right)

print("Estimated Projection Matrices:")
print("P_front:\n", P_front)
print("P_left:\n", P_left)
print("P_right:\n", P_right)

num_joints = skel_front.shape[0]
reconstructed_3d = np.zeros((n_frames_common, num_joints, 3))
for i in range(n_frames_common):
    for j in range(num_joints):
        p_f = skel_front[j, i, :]
        p_l = skel_left[j, i, :]
        p_r = skel_right[j, i, :]
        X_3d = triangulate_point([p_f, p_l, p_r], [P_front, P_left, P_right])
        reconstructed_3d[i, j, :] = X_3d

# ----- 3D 시각화 (예시: 한 프레임) -----
skeleton_edges = [(0,1), (1,2), (2,3), (3,4),
                  (5,7), (7,9),   # Left arm
                  (6,8), (8,10),  # Right arm
                  (11,13), (13,15),# Left leg
                  (12,14), (14,16)]# Right leg
# (추가적으로 얼굴이나 상체 연결선은 데이터셋에 맞게 추가 가능)

frame_idx = 0
pts = reconstructed_3d[frame_idx, :, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='r', s=50)

# 연결선 그리기
for (i, j) in skeleton_edges:
    ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], [pts[i,2], pts[j,2]], 'b-', lw=2)

# 각 관절에 대해 label2key 정보를 텍스트로 추가
for j in range(pts.shape[0]):
    ax.text(pts[j,0], pts[j,1], pts[j,2], label2key[j], fontsize=9)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Triangulated 3D Skeleton with Labels (Frame {})'.format(frame_idx))
plt.show()

# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from scipy.linalg import lstsq
# from mpl_toolkits.mplot3d import Axes3D

# # ----- 데이터 로드 함수 -----
# def load_2d_data(path):
#     """
#     pkl 파일에서 2D 관절 좌표(첫 두 값)와 confidence를 로드.
#     data['data']와 data['conf']는 key가 '0'부터 '16'까지인 dictionary라고 가정.
#     """
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     # 2D 좌표 (x, y)
    
#     print("관절 키들:", list(data['data'].keys()))
#     skeleton = []
#     for key in sorted(data['data'].keys(), key=lambda x: int(x)):
#         pts = np.array(data['data'][key])[:, :2]  # (n_frames, 2)
#         skeleton.append(pts)
#     skeleton = np.stack(skeleton, axis=0)  # (num_joints, n_frames, 2)
    
#     conf = []
#     for key in sorted(data['conf'].keys(), key=lambda x: int(x)):
#         c = np.array(data['conf'][key])  # (n_frames,)
#         conf.append(c)
#     conf = np.stack(conf, axis=0)  # (num_joints, n_frames)
#     return skeleton, conf

# # ----- 초기 3D 추정 (단순 모델) -----
# def initial_reconstruct_3d(skel_front, skel_left, skel_right):
#     """
#     단순 모델: 
#       X = front의 x,
#       Y = (front.y + left.y + right.y)/3,
#       Z = (left의 x - right의 x)/2
#     각 뷰의 2D 좌표(skel_front, skel_left, skel_right)는 (num_joints, n_frames, 2) 형태.
#     반환: (num_joints, n_frames, 3) 형태의 초기 3D 추정.
#     """
#     num_joints, n_frames, _ = skel_front.shape
#     recon = np.zeros((num_joints, n_frames, 3))
#     for j in range(num_joints):
#         for i in range(n_frames):
#             X = skel_front[j, i, 0]
#             Y = (skel_front[j, i, 1] + skel_left[j, i, 1] + skel_right[j, i, 1]) / 3.0
#             Z = (skel_left[j, i, 0] - skel_right[j, i, 0]) / 2.0
#             recon[j, i, :] = np.array([X, Y, Z])
#     return recon

# # ----- DLT를 통한 카메라 프로젝션 행렬 추정 -----
# def aggregate_correspondences(initial_3d, skeleton_2d, conf, threshold):
#     """
#     초기 3D 추정(initial_3d): (num_joints, n_frames, 3)
#     2D 관측치(skeleton_2d): (num_joints, n_frames, 2)
#     conf: (num_joints, n_frames)
#     threshold: confidence 임계값
#     고신뢰도 관절에 대한 대응 (X, p)를 모아, X는 4차원 동차좌표.
#     """
#     num_joints, n_frames, _ = skeleton_2d.shape
#     X_list, p_list = [], []
#     for j in range(num_joints):
#         for i in range(n_frames):
#             if conf[j, i] >= threshold:
#                 X = initial_3d[j, i, :]
#                 X_h = np.hstack([X, 1])  # (4,)
#                 p = skeleton_2d[j, i, :]  # (2,)
#                 X_list.append(X_h)
#                 p_list.append(p)
#     return np.array(X_list), np.array(p_list)

# def estimate_projection_matrix(X, p):
#     """
#     X: (N, 4) 동차좌표, p: (N, 2) 관측치.
#     p = P * X, P는 (2, 4)
#     선형 최소제곱법으로 P를 추정.
#     """
#     # 각 대응에 대해 두 개의 방정식을 세움.
#     N = X.shape[0]
#     A = []
#     b = []
#     for i in range(N):
#         X_i = X[i, :]
#         u, v = p[i, 0], p[i, 1]
#         # u = P[0,:] * X_i, v = P[1,:] * X_i
#         A.append(X_i)
#         b.append(u)
#         A.append(X_i)
#         b.append(v)
#     A = np.array(A)  # (2N, 4)
#     b = np.array(b)  # (2N,)
#     # 독립적으로 각 행을 추정
#     A_u = A[0::2]
#     b_u = b[0::2]
#     P0, _, _, _ = lstsq(A_u, b_u)
#     A_v = A[1::2]
#     b_v = b[1::2]
#     P1, _, _, _ = lstsq(A_v, b_v)
#     P = np.vstack([P0, P1])  # (2, 4)
#     return P

# # ----- 삼각측량 (Triangulation) -----
# def triangulate_point(p_list, P_list):
#     """
#     여러 뷰에서 관측된 2D 점들과 해당 뷰의 프로젝션 행렬 P를 이용해,
#     선형 삼각측량으로 3D 점을 추정.
#     p_list: list of 2D 점 (각각 shape (2,))
#     P_list: list of 해당 뷰의 2x4 행렬 P
#     각 뷰에 대해, 먼저 P를 3x4 행렬로 확장한 후, 다음 방정식을 만듭니다:
#         u * Q[2,:] - Q[0,:] = 0
#         v * Q[2,:] - Q[1,:] = 0
#     여기서 Q = [P; [0,0,0,1]] 입니다.
#     """
#     A = []
#     for p, P in zip(p_list, P_list):
#         # 2x4 행렬 P를 3x4로 확장: 임의의 3번째 행을 [0,0,0,1]로 추가
#         Q = np.vstack([P, np.array([0, 0, 0, 1])])
#         u, v = p[0], p[1]
#         A.append(u * Q[2, :] - Q[0, :])
#         A.append(v * Q[2, :] - Q[1, :])
#     A = np.array(A)
#     U, S, Vt = np.linalg.svd(A)
#     X_h = Vt[-1, :]
#     X_h = X_h / X_h[-1]
#     return X_h[:3]


# # ----- 메인 처리 -----
# # 데이터 파일 경로 (세 뷰)
# front_path = './output/1-frontcut/1-frontcut.pkl'
# left_path  = './output/1-leftcut/1-leftcut.pkl'
# right_path = './output/1-rightcut/1-rightcut.pkl'

# # 각 뷰의 2D 데이터를 로드
# skel_front, conf_front = load_2d_data(front_path)
# skel_left, conf_left   = load_2d_data(left_path)
# skel_right, conf_right = load_2d_data(right_path)

# # 동기화를 위해 각 뷰의 프레임 수의 최소값 사용
# n_frames_common = min(skel_front.shape[1], skel_left.shape[1], skel_right.shape[1])
# skel_front = skel_front[:, :n_frames_common, :]
# skel_left  = skel_left[:, :n_frames_common, :]
# skel_right = skel_right[:, :n_frames_common, :]
# conf_front = conf_front[:, :n_frames_common]
# conf_left  = conf_left[:, :n_frames_common]
# conf_right = conf_right[:, :n_frames_common]

# # 초기 3D 추정 (단순 모델)
# initial_3d = initial_reconstruct_3d(skel_front, skel_left, skel_right)  # shape: (num_joints, n_frames, 3)

# # 각 뷰별로 고신뢰도 대응 집합 추출 (임계값 설정, 예: 0.8)
# threshold = 0.8
# X_front, p_front = aggregate_correspondences(initial_3d, skel_front, conf_front, threshold)
# X_left,  p_left  = aggregate_correspondences(initial_3d, skel_left, conf_left, threshold)
# X_right, p_right = aggregate_correspondences(initial_3d, skel_right, conf_right, threshold)

# # 각 뷰별 프로젝션 행렬 추정 (2x4)
# P_front = estimate_projection_matrix(X_front, p_front)
# P_left  = estimate_projection_matrix(X_left, p_left)
# P_right = estimate_projection_matrix(X_right, p_right)

# print("Estimated Projection Matrices:")
# print("P_front:\n", P_front)
# print("P_left:\n", P_left)
# print("P_right:\n", P_right)

# # 이제 각 프레임, 각 관절에 대해 세 뷰의 관측치와 추정된 P들을 사용해 3D 점 삼각측량
# num_joints = skel_front.shape[0]
# reconstructed_3d = np.zeros((n_frames_common, num_joints, 3))
# for i in range(n_frames_common):
#     for j in range(num_joints):
#         p_f = skel_front[j, i, :]
#         p_l = skel_left[j, i, :]
#         p_r = skel_right[j, i, :]
#         # 각 뷰의 2D 관측치를 리스트로 구성하고, 동일한 P들을 사용
#         X_3d = triangulate_point([p_f, p_l, p_r], [P_front, P_left, P_right])
#         reconstructed_3d[i, j, :] = X_3d

# # ----- 3D 시각화 (예시: 한 프레임) -----
# # 간단한 skeleton connectivity (예시; 데이터셋에 맞게 수정 필요)
# skeleton_edges = [(0,1), (1,2), (2,3), (3,4),
#                   (1,5), (5,6), (6,7),
#                   (1,8), (8,9), (9,10),
#                   (8,11), (11,12), (12,13),
#                   (0,14), (14,15), (15,16)]

# frame_idx = 0
# pts = reconstructed_3d[frame_idx, :, :]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='r', s=50)
# for (i, j) in skeleton_edges:
#     ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], [pts[i,2], pts[j,2]], 'b-', lw=2)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Triangulated 3D Skeleton (Frame {})'.format(frame_idx))
# plt.show()
