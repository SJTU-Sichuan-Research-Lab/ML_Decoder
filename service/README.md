## Run in local environment
```bash
python infer.py  --model-name=tresnet_l --model-path=./models/tresnet_l_COCO__448_90_0.pth --pic-path=./data/test.jpg --image-size=448
```

## Docker Build
```bash
docker build -f service/Dockerfile -t mld:v1 .
```


## Develop
Mount the whole project into the container
```bash
docker rm -f test && docker run -it --gpus all -v "C:\Users\ycwei\PycharmProjects\nas\ML_Decoder:/app/tmp" -v "C:\Users\ycwei\PycharmProjects\nas\ML_Decoder\service\models:/app/models" -v "C:\Users\ycwei\PycharmProjects\nas\ML_Decoder\service\data:/app/data" --name test mld python tmp/infer.py  --model-name=tresnet_l --model-path=models/tresnet_l_COCO__448_90_0.pth --pic-path=data/test.jpg --image-size=448
```


## Run
app(python)
```bash
docker rm -f test
docker run -it --gpus all -v "C:\Users\ycwei\PycharmProjects\nas\ML_Decoder\service\models:/app/models" -v "C:\Users\ycwei\PycharmProjects\nas\ML_Decoder\service\data:/app/data" --name test mld:v1 python infer.py  --model-name=tresnet_l --model-path=models/tresnet_l_COCO__448_90_0.pth --pic-path=data/test.jpg --image-size=448
```
bash
```bash
docker rm -f test
docker run -it --gpus all -v "C:\Users\ycwei\PycharmProjects\nas\ML_Decoder\service\models:/app/models" -v "C:\Users\ycwei\PycharmProjects\nas\ML_Decoder\service\data:/app/data" --name test mld:v1 bash
```
