from scipy.sparse import csr_matrix

def load_data(data):
    
    path_train = './dataset/' + str(data) + '/train.txt'
    path_test = './dataset/' + str(data) + '/test.txt'

    users_list_train = [] 
    items_list_train = [] 

    with open(path_train, 'r') as f:
        for line in f:
            users_list_train.append(int(line.split()[0]))
            items_list_train.append(list(map(int,line.split()[1:])))

    num_users_train = len(users_list_train)

    users_list_test = [] 
    items_list_test = [] 

    with open(path_test, 'r') as f:
        for line in f:
            users_list_test.append(int(line.split()[0]))
            items_list_test.append(list(map(int,line.split()[1:])))

    num_users_test = len(users_list_test)
    
    indptr_train = [0]
    indices_train = []
    data_train = []

    for i in range(num_users_train):
        for item in items_list_train[i]:
            indices_train.append(item)
            data_train.append(1)

        indptr_train.append(len(indices_train))

    indptr_test = [0]
    indices_test = []
    data_test = []


    for i in range(num_users_test):
        for item in items_list_test[i]:
            indices_test.append(item)
            data_test.append(1)

        indptr_test.append(len(indices_test))

    # To ensure train and test matrix shape equal
    if max(indices_train) > max(indices_test):
        indices_test.append(max(indices_train))
        data_test.append(0)
    else:
        indices_train.append(max(indices_test))
        data_train.append(0)

    X_train = csr_matrix((data_train, indices_train, indptr_train), dtype=int)
    X_train.eliminate_zeros()

    X_test = csr_matrix((data_test, indices_test, indptr_test), dtype=int)
    X_test.eliminate_zeros()

    return X_train, X_test, num_users_test, items_list_test, users_list_test
    
