import taichi as ti
import sys

@ti.data_oriented
class VelocityGradientComputer:
    def __init__(self, num_particles, dim = 3, k = 8):
        
        self.n = num_particles
        # self.dim = ti.field(ti.i32, shape=())
        # self.k = ti.field(ti.i32, shape=())
        self.dim = dim
        self.k = k # 最近邻个数
        # self.dx = dx
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n[None], needs_grad=True)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n[None], needs_grad = True)

        # n_particles = int(self.n[None])
        # self.sample_idx = ti.field(dtype = ti.i32, shape = (n_particles, 50, self.dim))

        self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n[None], needs_grad = True)

        self.neighbors = ti.field(dtype=ti.i32, shape=(self.n[None], self.k))
        self.neighbor_distances = ti.field(dtype=ti.f32, shape=(self.n[None], self.k))
    

        self.p_list = ti.Vector.field(self.dim + 1, dtype=ti.f32, shape=(self.k,))
        self.w_list = ti.field(dtype=ti.f32, shape=self.k)

        
    @ti.kernel
    def set_particles(self, positions: ti.types.ndarray(), velocities: ti.types.ndarray()):
        for i in range(self.n[None]):
            for d in ti.static(range(self.dim)):
                self.x[i][d] = positions[i, d]
                self.v[i][d] = velocities[i, d]
    @ti.kernel
    def find_neighbors(self):
        # K = ti.static(self.k)  

        for i in range(self.n[None]):
            min_dists = ti.Vector([1e16 for _ in ti.static(range(self.k))], dt=ti.f32)
            min_ids = ti.Vector([-1 for _ in ti.static(range(self.k))], dt=ti.i32)
            # min_dists_next = ti.Vector([1e13 for _ in range(K)], dt=ti.f32)
            # min_ids_next = ti.Vector([-1 for _ in range(K)], dt=ti.i32)

            xi = self.x[i]
            # xi_next = self.x_next[i]
            for j in range(self.n[None]):
                if i != j:
                    xj = self.x[j]
                    # xj_next = self.x_next[j]
                    dist = (xi - xj).norm()
                    # dist_next = (xi_next - xj_next).norm()
                    inserted = False
                    # inserted_next = False
                    for k in ti.static(range(self.k)):
                        if not inserted and dist < min_dists[k]:
                            for s in ti.static(list(reversed(range(k + 1, self.k)))):
                                min_dists[s] = min_dists[s - 1]
                                min_ids[s] = min_ids[s - 1]
                            min_dists[k] = dist
                            min_ids[k] = j
                            inserted = True
                        '''
                        if not inserted_next and dist_next < min_dists_next[k]:
                            for s in ti.static(list(reversed(range(k + 1, K)))):
                                min_dists_next[s] = min_dists_next[s - 1]
                                min_ids_next[s] = min_ids_next[s - 1]
                            min_dists_next[k] = dist_next
                            min_ids_next[k] = j
                            inserted_next = True  
                        '''
            for k in ti.static(range(self.k)):
                self.neighbors[i, k] = min_ids[k]
                self.neighbor_distances[i, k] = min_dists[k]
                # self.neighbors_next[i, k] = min_ids_next[k]
                # self.neighbor_distances_next[i, k] = min_dists_next[k]

    @ti.func
    def cubic_bspline_weight(self, x):
        abs_x = ti.abs(x)
        val = 0.0
        if abs_x < 1:
            val = 2.0 / 3.0 - abs_x**2 + 0.5 * abs_x**3
        elif abs_x < 2:
            val = 1.0 / 6.0 * (2 - abs_x)**3
        else:
            val = 0.0
        return val
    @ti.func
    def rbf_weight(self, r: ti.f32) -> ti.f32:
    # Gaussian RBF 核函数，权重计算公式为 exp(-r^2 / (2 * epsilon^2))
        # epsilon = ti.cast(0.06, ti.f32)  # 可以根据需要调整，控制核函数的宽度
        return ti.exp(-r * r / (2 * 0.06 * 0.06))
    @ti.func
    def wendland_kernel(self, r):
        q = r / 0.85
        if q < 1.0:
            return (1.0 - q)**4 * (1.0 + 4.0 * q)
        else:
            return 0.0
    @ti.func
    def compute_mls_weights(self, ind):

        A = ti.Matrix.zero(ti.f32, self.dim + 1, self.dim + 1)
    
        # 插值点处的基函数 P(z - x)，这里 z = x，因此 p0 = [1, 0, 0, ..., 0]
        p0 = ti.Vector.zero(ti.f32, self.dim + 1)
        # p0[0] = 1.0
        p0[0] = ti.cast(1, ti.f32)
        weights = ti.Vector.zero(ti.f32, self.k)
        for kk in ti.static(range(self.k)):
            j = self.neighbors[ind, kk]
            # if j == -1:
            #     continue

            dx = self.x[j] - self.x[ind]
            
            r = dx.norm()
            w = self.rbf_weight(r)

            # 构建 P(x_j - x)
            p = ti.Vector.zero(ti.f32, self.dim + 1)
            # p[0] = 1.0
            p[0] = ti.cast(1, ti.f32)
            for d in ti.static(range(self.dim)):
                p[d + 1] = dx[d]
            self.p_list[kk] = p
            self.w_list[kk] = w

            # A += w * p * p^T
            A += w * p.outer_product(p)

        A_inv = A.inverse()
        # w_sum = 0.
        w_sum =ti.cast(0, ti.f32)
        # x_sum =  ti.Vector.zero(ti.f32, self.dim)
        for kk in ti.static(range(self.k)):
            weights[kk] = self.w_list[kk] * (p0 @ A_inv @ self.p_list[kk])
            w_sum += weights[kk]

        for kk in ti.static(range(self.k)):
            weights[kk] /= w_sum

        # print(">>", x_sum, xi)
        return weights
        
    '''
    @ti.kernel
    def compute_velocity_gradient(self):
        for i in range(self.n):
            xi = self.x[i]
            C_i = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            inv_dx = 1.0 / (self.neighbor_distances[i, 0] )
            vi = self.v[i]
            # inv_dx = 1.0 / self.dx[None]
            w_sum = 0.
            for k in range(self.k):
                j = self.neighbors[i, k]
                if j != -1:
                    xj = self.x[j]
                    vj = self.v[j]
                    dx = xj - xi
                    dv = vj - vi
                    dist = dx.norm()

                    weight = 1.0
                    for d in ti.static(range(self.dim)):
                        weight *= self.cubic_bspline_weight(dx[d] / ti.abs((self.x[self.neighbors[i, 0]] - self.x[i])[d] ))
                        # weight *= self.cubic_bspline_weight(dx[d] * inv_dx)
                        dx[d] = dx[d] / ti.abs((self.x[self.neighbors[i, 0]] - self.x[i])[d]) 
                        # dx[d] = dx[d] * inv_dx

                    # weight = 1. / ti.abs(dist)
                    w_sum += weight
                    C_i += 4 * weight * vj.outer_product(dx) * inv_dx

            C_i /= w_sum
            # print(">>", C_i)
            self.C[i] = C_i
    
    @ti.func
    def compute_mls_weights_next(self, ind):
        xi = self.x_next[ind]


        A = ti.Matrix.zero(ti.f32, self.dim + 1, self.dim + 1)

        # 插值点处的基函数 P(z - x)，这里 z = x，因此 p0 = [1, 0, 0, ..., 0]
        p0 = ti.Vector.zero(ti.f32, self.dim + 1)
        p0[0] = 1.0
        weights = ti.Vector.zero(ti.f32, self.k)
        for kk in range(self.k):
            j = self.neighbors_next[ind, kk]
            if j == -1:
                continue

            dx = self.x_next[j] - xi
            
            r = dx.norm()
            w = self.rbf_weight(r)

            # 构建 P(x_j - x)
            p = ti.Vector.zero(ti.f32, self.dim + 1)
            p[0] = 1.0
            for d in range(self.dim):
                p[d + 1] = dx[d]
            self.p_list[kk] = p
            self.w_list[kk] = w

            # A += w * p * p^T
            A += w * p.outer_product(p)

        A_inv = A.inverse()
        w_sum = 0.
        x_sum =  ti.Vector.zero(ti.f32, self.dim)
        for kk in range(self.k):
            weights[kk] = self.w_list[kk] * (p0 @ A_inv @ self.p_list[kk])
            w_sum += weights[kk]

        for kk in range(self.k):
            weights[kk] /= w_sum

        # print(">>", x_sum, xi)
        return weights
    
    @ti.kernel
    def compute_velocity_gradient_next(self):
        for i in range(self.n):
            weight = self.compute_mls_weights_next(i)
            # weight_next = self.compute_mls_weights_next(i)
            xi = self.x_next[i]
            # xi_next = self.x_next[i]
            # C_i = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            B = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            D = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            vi = self.v_next[i]
            x_sum =  ti.Vector.zero(ti.f32, self.dim)
            dist_sum = 0.
            for k in range(self.k):
                j = self.neighbors_next[i, k]
  
                xj = self.x_next[j]

                dx = xj - xi
                dist = dx.norm()
                dist_sum += dist
            dist_sum /= self.k
            for k in range(self.k):
                j = self.neighbors_next[i, k]
  
                xj = self.x_next[j]
                x_sum += weight[k] * xj
                vj = self.v_next[j] - vi

                
                dx = (xj - xi)
                
                
                
                B += weight[k] * vj.outer_product(dx)
                # B += (weight[k] * vj.outer_product(dx))
                D += (weight[k] * dx.outer_product(dx))
                
            self.C_next[i] = B @ D.inverse() 
    '''
    @ti.kernel
    def compute_velocity_gradient(self):
        for i in range(self.n[None]):
            '''
            weight = self.compute_mls_weights(i)
            # weight_next = self.compute_mls_weights_next(i)
            xi = self.x[i]
            # xi_next = self.x_next[i]
            # C_i = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            B = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            D = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            vi = self.v[i]
            x_sum =  ti.Vector.zero(ti.f32, self.dim)

            for k in ti.static(range(self.k)):
                j = self.neighbors[i, k]
  
                xj = self.x[j]
                x_sum += weight[k] * xj
                vj = self.v[j] - vi

                
                dx = (xj - xi)
                
                
                
                B += weight[k] * vj.outer_product(dx)
                # B += (weight[k] * vj.outer_product(dx))
                D += (weight[k] * dx.outer_product(dx))
            '''
            # self.C[i] = B @ D.inverse() 
            self.C[i] = self.ransac_velocity_gradient(i)
            # print(i, self.C[i])

    @ti.kernel
    def get_velocity_gradients(self):
        for i in ti.static(range(self.n[None])):
            print(f"Particle {i}, C = {self.C[i]}")
    '''
    @ti.func
    def ransac_velocity_gradient(self, ind):

        xi = self.x[ind]
        vi = self.v[ind]
        
        best_inlier_count = 0
        C_best = ti.Matrix.zero(ti.f32, self.dim, self.dim)


        for ii in range(self.k):
            for jj in range(ii+1, self.k):
                for kk in range(jj+1, self.k):
                    B = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                    D = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                        
                    dx_1 = self.x[self.neighbors[ind, ii]] - xi
                    dv_1 = self.v[self.neighbors[ind, ii]] - vi
                    B += dv_1.outer_product(dx_1)
                    D += dx_1.outer_product(dx_1)

                    dx_2 = self.x[self.neighbors[ind, jj]] - xi
                    dv_2 = self.v[self.neighbors[ind, jj]] - vi
                    B += dv_2.outer_product(dx_2)
                    D += dx_2.outer_product(dx_2)

                    dx_3 = self.x[self.neighbors[ind, kk]] - xi
                    dv_3 = self.v[self.neighbors[ind, kk]] - vi
                    B += dv_3.outer_product(dx_3)
                    D += dx_3.outer_product(dx_3)


                    D += 1e-6 * ti.Matrix.identity(ti.f32, self.dim)
                    C_trial = B @ D.inverse()


                    inlier_count = 0
                    threshold = 0.02
                    for k in range(self.k):
                        j = self.neighbors[ind, k]
                        dx = self.x[j] - xi
                        dv = self.v[j] - vi
                        dv_pred = C_trial @ dx
                        err = (dv - dv_pred).norm()
                        if err < threshold:
                            inlier_count += 1


                    if inlier_count > best_inlier_count:
                        best_inlier_count = inlier_count
                        C_best = C_trial

        return C_best
    '''
    @ti.func
    def ransac_velocity_gradient(self, ind):
        xi = self.x[ind]
        vi = self.v[ind]

        best_inlier_count = 0
        C_best = ti.Matrix.zero(ti.f32, self.dim, self.dim)

        for ii in range(self.k):
            for jj in range(ii + 1, self.k):
                for kk in range(jj + 1, self.k):
                    B = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                    D = ti.Matrix.zero(ti.f32, self.dim, self.dim)

                    # 依次处理 ii, jj, kk
                    neighbor_id = self.neighbors[ind, ii]
                    dx = ti.stop_grad(self.x[neighbor_id] - xi)
                    dv = ti.stop_grad(self.v[neighbor_id] - vi)
                    B += dv.outer_product(dx)
                    D += dx.outer_product(dx)

                    neighbor_id = self.neighbors[ind, jj]
                    dx = ti.stop_grad(self.x[neighbor_id] - xi)
                    dv = ti.stop_grad(self.v[neighbor_id] - vi)
                    B += dv.outer_product(dx)
                    D += dx.outer_product(dx)

                    neighbor_id = self.neighbors[ind, kk]
                    dx = ti.stop_grad(self.x[neighbor_id] - xi)
                    dv = ti.stop_grad(self.v[neighbor_id] - vi)
                    B += dv.outer_product(dx)
                    D += dx.outer_product(dx)

                    D += 1e-6 * ti.Matrix.identity(ti.f32, self.dim)
                    C_trial = B @ D.inverse()

                    # inlier_count 内部不追踪梯度
                    inlier_count = 0
                    threshold = 0.02
                    for k_idx in range(self.k):
                        j = self.neighbors[ind, k_idx]
                        dx = ti.stop_grad(self.x[j] - xi)
                        dv = ti.stop_grad(self.v[j] - vi)
                        dv_pred = C_trial @ dx
                        err = (dv - dv_pred).norm()
                        if err < threshold:
                            inlier_count += 1

                    if inlier_count > best_inlier_count:
                        best_inlier_count = inlier_count
                        C_best = C_trial  # C_best 可参与梯度

        return C_best
