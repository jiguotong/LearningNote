"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/6/27
@Description:   使用opencv进行图像拼接的案例
"""""
import cv2
import sys

def stitch_images(images):
    # 创建 Stitcher 对象（OpenCV 3.x 使用 cv2.createStitcher()）
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

    # 执行拼接
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("图像拼接成功 ✅")
        return stitched
    else:
        print(f"图像拼接失败 ❌ 状态码: {status}")
        return None

if __name__ == "__main__":
    # 读取多张图像（顺序重要：左到右）
    img1 = cv2.imread("sift1.png")
    img2 = cv2.imread("sift2.png")

    if img1 is None or img2 is None:
        print("图像加载失败，请检查路径")
        sys.exit()

    # 进行拼接
    result = stitch_images([img1, img2])

    # 显示或保存结果
    if result is not None:
        cv2.imshow("Stitched Image", result)
        cv2.imwrite("stitched_output.jpg", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
