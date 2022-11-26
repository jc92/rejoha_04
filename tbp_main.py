import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def upsample(new_line):
#     print('Original Dimensions : ',new_line.shape)
    scale_percent = 150 # percent of original size
    width = int(new_line.shape[1] * scale_percent / 100)
    height = int(new_line.shape[0])
    dim = (width, height)
    # resize image
    resized = cv2.resize(new_line, dim, interpolation = cv2.INTER_AREA)
#     print('Resized Dimensions : ',resized.shape)
    return np.array([x[:width] for x in resized])


def run(file_name,pixel_v_coordinate,upsample_flag,slice_width):
    file_loc = f'./01_input/{file_name}'

    video = cv2.VideoCapture(file_loc)
    # w_video = int(video.get(cv2.CAP_PROP_FRAME_WIDTH ))
    h_video = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    frames_counter = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # output_frame = Image.new("RGB", (frames_counter,h))
    output_frame = []
    with tqdm(total=frames_counter) as pbar:
        while True:
            read, frame = video.read()
            pbar.update(1)
            if not read:
                break
            # print(frame.shape)
            h,w,c = frame.shape # todo clean
            new_line = np.array([x[pixel_v_coordinate:(pixel_v_coordinate+slice_width)][0][::-1] for x in frame]).astype('uint8').reshape(h, -1, c) # [0] is kinda hacky
            if upsample_flag:
                new_line = upsample(new_line) # upsample not tested prob faulty
            for _ in range(slice_width):
                output_frame.append(new_line)
    # formatting, hacky just to make it work
    output_frame_array = np.array(output_frame[::-1])
    output_frame_array_reshape = output_frame_array.reshape(-1, h_video, 3)
    pil_img = Image.fromarray(output_frame_array_reshape).convert('RGB')
    pil_img_final = pil_img.transpose(Image.ROTATE_90).rotate(180)

    if upsample_flag:
        pil_img_final.save(f"./02_output/{file_name.split('.')[0]}_upsample_{slice_width}.png", "PNG")
    else:
        pil_img_final.save(f"./02_output/{file_name.split('.')[0]}.png", "PNG")

if __name__ == '__main__':
    # file_name = 'klara_4k_55fps.MOV'
    # file_name = 'fastnacht_short.mp4'
    file_name = 'test2.MOV'
    # file_name = 'test3.MOV'
    pixel_v_coordinate = 100 # vertical pixel line to use for collating
    upsample_flag = True
    slice_width = 2  # lowers horizontal res, set to 1 if upsample_flag set to false
    run(file_name,pixel_v_coordinate,upsample_flag,slice_width)