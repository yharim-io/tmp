from clipcap.config import Config
from clipcap.layer import ClipCaptionModel, MappingType
from clipcap.engine.decode import image_to_text, device 

model = ClipCaptionModel(
	prefix_length=Config.model.prefix_length,
	clip_length=Config.model.prefix_length,
	prefix_size=Config.model.clip_dim,
	num_layers=Config.model.num_layers,
	mapping_type=MappingType.MLP
)

# 在这里加载你训练好的权重
# model_weights_path = Config.path.root / 'data/clipcap/coco/037.pt'
# model.load_state_dict(
# 	torch.load(
# 		model_weights_path,
# 		map_location=torch.device('cpu'),
# 		weights_only=True
# 	)
# )

model = model.to(device)
model.eval()

for i in range(1, 9):
	
	image_file = Config.path.root/f'data/images/{i}.jpg'

	text = image_to_text(
		clipcap_model=model,
		image_path=image_file
	)

	print(f"Image {i}.jpg: {text}")

