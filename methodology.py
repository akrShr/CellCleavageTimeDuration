import torch
import detect
import os
import shutil
import cv2
import glob
import re
import pytesseract
import xlsxwriter


input_folder = './resources/data'
results_folder = './runs/detect/exp'
excelSheet_path = './resources/excelSheets/Cleavage_StartTime.xlsx'
videos_pth = './resources/videos/*.avi'
delKeys =['t0','t1']


print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

def ocr_core(filename):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(filename,lang='eng', config=custom_config)
    value = text.split('\n')[0].replace(" ","").split('h')[0]
    value = ''.join(filter(str.isdigit, value))
    if value.find('.') == -1:
       value = float(value)/10
    return format(float(value),".1f")

def get_frame(image_path):
    frame = cv2.imread(image_path)
    height, width, channel = frame.shape
    frame[:int(0.9375 * (height)), int(0.1339 * (width)):, :] = 0
    frame[:int(0.9375 * (height)), :int(0.8482 * (width)), :] = 0
    frame[int(0.9375 * (height)):, :int(0.1339 * (width)), :] = 0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    i = 499
    bottom = False
    while i > 440:
        j = 499
        while j > 440:
            if gray_frame[i, j] > 200:
                if not bottom:
                    v = (i, j)
                    bottom = True
                else:
                    min_j = j
            j = j - 1
        if bottom:
            u = (i, min_j)
            break
        i = i - 1
    cropped = gray_frame[475:u[0], u[1]:v[1]]
    return ocr_core(cropped)

def detect_boxes():
    opt = detect.parse_opt()
    detect.main(opt)


def create_folders():
    try:
        os.makedirs(input_folder+'/ODFrames')
        os.makedirs(input_folder+'/Frames')
    except OSError:
        print("Creation of the directory %s failed" % input_folder)
    else:
        print("Successfully created the directory %s" % input_folder)

def remove_folders():
    try:
        shutil.rmtree(input_folder)
        shutil.rmtree(results_folder)
    except OSError as e:
        print("Deletion of the directory %s failed" % e.filename)
    else:
        print("Successfully deleted the directories")

def extract_frames(vid_id):
    video = cv2.VideoCapture(vid_id)
    i = 1
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == False:
            break
        resizedframe = cv2.resize(frame, (416, 416))
        cv2.imwrite(input_folder + '/Frames/' + str(i) + '.jpg', frame)
        cv2.imwrite(input_folder + '/ODFrames/' +str(i) + '.jpg', resizedframe)
        i += 1
    video.release()


def count_boxes():
    framesDict = {'t0': -1, 't1': -1}
    cellsConf=[.50,.50,.50,.50,.50,.60,.65,.70,.80]
    cellConf=cellsConf.pop()
    labels=glob.glob(results_folder+'/labels/*.txt')
    labels.sort(key=lambda f: int(re.sub('\D', '', f)))
    for label in labels:
        cells=0
        texts = open(label).readlines()
        if len(framesDict)>2 and len(texts)<2:
            conf_str = texts[0].split(" ")[-1].split("\n")[0]
            conf = float("{:.2f}".format(float(conf_str)))
            if texts[0][0] == '2' and conf > .90:
                fKey = 'tM'
            elif texts[0][0] == '0' and conf > .90:
                fKey= 'tB'
            if not fKey in framesDict:
                framesDict[fKey] = label.split("/")[-1].split(".txt")[0]
        else:
            for text in texts:
                conf_str = text.split(" ")[-1].split("\n")[0]
                conf = float("{:.2f}".format(float(conf_str)))
                if text[0] == '1' and conf>=cellConf:
                    cells +=1
            if len(cellsConf)>0:
                fKey = 't' + str(cells)
                if not fKey in framesDict:
                    framesDict[fKey]=label.split("/")[-1].split(".txt")[0]
                    cellConf = cellsConf.pop()
    for key in delKeys:
        if key in framesDict:
            del framesDict[key]
    return framesDict

def generate_time_annotations():
    excel_header = ['Project_ID', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 'tM', 'tB']
    workbook = xlsxwriter.Workbook(excelSheet_path)
    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 12)
    bold = workbook.add_format({'bold': True})
    col_idx = 0
    for col in excel_header:
        worksheet.write(0, col_idx, col, bold)
        col_idx += 1
    videos = sorted(glob.glob(videos_pth))
    i = 1
    for video in videos:
        create_folders()
        extract_frames(video)
        detect_boxes()
        framesDict = count_boxes()
        worksheet.write(i, 0, video.split('/')[-1].split('.a')[0])
        for key, value in framesDict.items():
            col_idx = excel_header.index(key)
            timeStmp = get_frame(input_folder + '/Frames/' + value + '.jpg')
            worksheet.write(i, col_idx, timeStmp)
        i += 1
        remove_folders()
    workbook.close()


if __name__ == "__main__":
    generate_time_annotations()



