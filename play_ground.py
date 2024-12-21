import scipy

mat_data = scipy.io.loadmat('/home/qjw/code/python_code/phase_field/FullData_ACequ_2D.mat')
phiN = mat_data['phiN']

print(f"Type of phiN: {type(phiN)}")
print(f"Shape of phiN: {phiN.shape}")

# Optionally, display part of the data
print(phiN)