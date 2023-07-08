# af = gen_assemble_fns(
#     do,
#     D,
#     h_x,
#     h_y,
#     h_z,
# ) 

# h_x = np.array([0.2])
# h_y = np.array([0.0])
# h_z = np.array([0.0])
# af = gen_assemble_fns(
#     do,
#     D,
#     h_x,
#     h_y,
#     h_z,
# ) 
# theta, M_0, M_x, M_y, M_z, Delta, omega_n = np.array([0.50]), np.array([0.40]), np.array([0.1]), np.array([0]), np.array([0]), np.array([1]), np.array([0.250])

# print(af['f0'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df0_dtheta'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df0_dM_x'](theta, M_0, M_x, M_y, M_z, Delta, omega_n).todense())
# print(af['f1'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df1_dtheta'](theta, M_0, M_x, M_y, M_z, Delta, omega_n).todense())
# print(af['df1_dM_x'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))


# print()
# h_x = np.array([0.0])
# h_y = np.array([0.2])
# h_z = np.array([0.0])
# af = gen_assemble_fns(
#     do,
#     D,
#     h_x,
#     h_y,
#     h_z,
# ) 
# theta, M_0, M_x, M_y, M_z, Delta, omega_n = np.array([0.50]),np.array([0.40]), np.array([0]), #np.array([0.1]), np.array([0]), np.array([1]), np.array([0.250])

# print(af['f0'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df0_dtheta'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df0_dM_y'](theta, M_0, M_x, M_y, M_z, Delta, omega_n).todense())
# print(af['f2'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df2_dtheta'](theta, M_0, M_x, M_y, M_z, Delta, omega_n).todense())
# print(af['df2_dM_y'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))


# print()
# h_x = np.array([0.0])
# h_y = np.array([0.0])
# h_z = np.array([0.2])
# af = gen_assemble_fns(
#     do,
#     D,
#     h_x,
#     h_y,
#     h_z,
# ) 
# theta, M_0, M_x, M_y, M_z, Delta, omega_n = np.array([0.50]),np.array([0.40]), np.array([0]), np.array([0.0]), np.array([0.1]), np.array([1]), np.array([0.250])

# print(af['f0'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df0_dtheta'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df0_dM_z'](theta, M_0, M_x, M_y, M_z, Delta, omega_n).todense())
# print(af['f3'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))
# print(af['df3_dtheta'](theta, M_0, M_x, M_y, M_z, Delta, omega_n).todense())
# print(af['df3_dM_z'](theta, M_0, M_x, M_y, M_z, Delta, omega_n))