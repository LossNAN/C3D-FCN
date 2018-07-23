usage:
	1>tensorflow 1.2.0
	
	2>pre_train sport1m where you can find https://www.dropbox.com/s/zvco2rfufryivqb/conv3d_deepnetA_sport1m_iter_1900000_TF.model?dl=0
	
	3>you should have your own train and test list such as we share in annotation
	
	4>you can use some sh in packege('list') to convert your video to frame with 5fps
	
	5>use python train_pool5.py train your model
	
	6>use python predict_c3d_fcn.py to get result
	
	7>use MAP-IOU by jupyter notebook to get mAP