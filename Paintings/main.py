from Module._database import *
from Module._network import *

# ALL NEEDED DIRECTIORIES
dir = 'D:/Uni/SEMESTERS/MS/II/MonographicComputation/LAB/MachineLearningLearning/Paintings/catalogMy.xlsx'
test_dir = 'D:/Uni/SEMESTERS/MS/II/MonographicComputation/LAB/MachineLearningLearning/Paintings/new.xlsx'
learnImDir = 'D:/Uni/SEMESTERS/MS/II/MonographicComputation/LAB/MachineLearningLearning/Paintings/Images'
testImDir = 'D:/Uni/SEMESTERS/MS/II/MonographicComputation/LAB/MachineLearningLearning/Paintings/ImagesTest'

train_cols = ['Images', 'FORM']
save_name = 'from_imagenet_FORM_2.h5'

# WE HAVE 1000 test images
testPd = createImageDatabase(test_dir, testImDir, False, howmany=2000)
# WE HAVE 10000 images in image
learnPd = createImageDatabase(dir, learnImDir, False, howmany=10000)

# create model
db = learnPd.loc[:, train_cols]
db['ID'] = db.groupby(train_cols[1]).ngroup()
print(db.head())
dic = (db.groupby(train_cols[1]).first())['ID'].to_dict()
print(dic)
types_num = len(dic)


#history = createModel(learnPd, learnImDir, save_name, train_cols, types_num, epo=10, batch=40)
#printHistory(history)

predict, label_map_test = testModel(testPd, testImDir, save_name, train_cols)
label_map_orig = check_orig_labels(learnPd, learnImDir, train_cols)
# we can see the differences in labeling
print(label_map_orig)
print(label_map_test)
# change the dictionary form to take the categories
final_dic = change_keys_from_org_to_new(label_map_orig, label_map_test)
print(final_dic)
# create new dataframe with good predictions
test_check = testPd.loc[:, train_cols]

max_index = np.argmax(predict, axis=1)
test_check['INDEX_PREDICTED'] = max_index
test_check['CATEGORY_PREDICTED'] = test_check['INDEX_PREDICTED'].map(lambda a: final_dic[a])
# test_check = test_check.drop('INDEX_PREDICTED')
good_checks = test_check[test_check['FORM'] == test_check['CATEGORY_PREDICTED']]
print(good_checks)
