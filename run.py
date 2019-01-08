import dataloader
from categoryclassifier import CategoryClassifier

training_data, validation_data = dataloader.split_data('D:\data\kakao_arena')

bcateid, mcateid, scateid, dcateid, model, brand, maker = zip(*training_data)

v_bcateid, v_mcateid, v_scateid, v_dcateid, v_model, v_brand, v_maker = zip(*validation_data)

classifier = CategoryClassifier()
classifier.training(model, bcateid, v_model, v_bcateid)
