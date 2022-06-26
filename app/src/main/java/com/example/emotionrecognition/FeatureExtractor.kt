package com.example.emotionrecognition

import android.content.Context
import android.graphics.*
import android.media.MediaMetadataRetriever
import android.util.Log
import android.widget.ImageView
import androidx.annotation.UiThread
import androidx.annotation.WorkerThread
import com.example.emotionrecognition.mtcnn.Box
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jvm.JvmMath.minD2
import org.jetbrains.kotlinx.multik.jvm.JvmMath.maxD2
import org.jetbrains.kotlinx.multik.jvm.JvmStatistics.meanD2
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.*
import java.nio.FloatBuffer
import java.util.*
import kotlin.time.ExperimentalTime


class FeatureExtractor(context: Context, feat_len: Int, filename: String) {
    companion object {
        private const val TAG = "Video detection"
        private var featureSize = 0

        @Throws(IOException::class)
        fun assetFilePath(context: Context, feat_len: Int, assetName: String): String {
            featureSize = feat_len
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }
            context.assets.open(assetName!!).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * featureSize)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        }

        @UiThread
        fun applyToUiAnalyzeImageResult(result: AnalysisResult?, width: Int, height: Int, mOverlayView: ImageView, isCamera: Boolean) {
            val emotion = result!!.mResults
            val tempBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val c = Canvas(tempBmp)
            val p = Paint()
            p.style = Paint.Style.STROKE
            p.isAntiAlias = true
            p.isFilterBitmap = true
            p.isDither = true
            p.color = Color.parseColor("#9FFFCB")
            p.strokeWidth = 8f
            val p_text = Paint()
            p_text.color = Color.WHITE
            p_text.style = Paint.Style.FILL
            p_text.color = Color.parseColor("#9FFFCB")
            p_text.textSize = 90f
            val bbox = result.box
            p.color = Color.parseColor("#9FFFCB")
            c.drawRect(bbox, p)
            if (isCamera) {
                c.drawText(emotion, bbox.right.toFloat(), Math.max(0, bbox.top - 40).toFloat(), p_text)
            }
            else {
                c.drawText(emotion, bbox.left.toFloat(), Math.max(0, bbox.top - 40).toFloat(), p_text)
            }

            mOverlayView.setImageBitmap(tempBmp)
        }
    }

    private var labels: ArrayList<String>? = null
    private var module: Module? = null

    class AnalysisResult(val box: Rect, val mResults: String, val width: Int, val height: Int)

    private fun loadLabels(context: Context) {
        val br: BufferedReader?
        labels = ArrayList()
        try {
            br = if (WelcomeActivity.isEngagement) {
                BufferedReader(InputStreamReader(context.assets.open("engagement_labels.txt")))
            } else {
                BufferedReader(InputStreamReader(context.assets.open("emotion_labels.txt")))
            }
            br.useLines { lines -> lines.forEach {
                val categoryInfo = it.trim { it <= ' ' }.split(":").toTypedArray()
                val category = categoryInfo[1]
                labels!!.add(category) } }
            br.close()
        } catch (e: IOException) {
            throw RuntimeException("Problem reading label file!", e)
        }
        Log.e(TAG, labels.toString())
    }

    private fun classifyFeaturesReg(res: FloatArray): String {
        val scores = mutableListOf<Float>()
        val score2class = mapOf(0.0 to 0, 0.33 to 1, 0.66 to 2, 0.99 to 3)
        for (i in 0 until Constants.COUNT_OF_FRAMES_PER_INFERENCE){
            if ((i+1)*featureSize <= res.size) {
                scores.addAll(res.sliceArray(featureSize*i until featureSize*(i+1)).toList())
            }
        }
        val features = mk.ndarray(mk[scores])
        val min = minD2(features, axis = 0).toList()
        val descriptor = min
        var score = EngagementActivity.clf?.predict(descriptor)
        Log.e(EngagementActivity.TAG, "Prediction score: $score")
        if (score!! > 1) {
            score = 1.0
        }
        else if (score < 0) {
            score = 0.0
        }

        val scoreAdj = 0.33*(Math.round(score!! /0.33))

        return labels!![score2class[scoreAdj]!!]
    }

    private fun classifyFeaturesClass(res: FloatArray): String {
        val scores = mutableListOf<Float>()
        for (i in 0 until Constants.COUNT_OF_FRAMES_PER_INFERENCE){
            if ((i+1)* featureSize <= res.size) {
                scores.addAll(res.sliceArray(featureSize*i until featureSize*(i+1)).toList())
            }
        }
        val features = mk.ndarray(mk[scores])
        val min = minD2(features, axis = 0).toList()
        val max = maxD2(features, axis = 0).toList()
        val mean: List<Float> = meanD2(features, axis = 0).toList().map { it.toFloat() }
        val std = mutableListOf<Float>()
        val rows = features.shape[0]
        for (i in 0 until featureSize) {
            std.add(calculateSD(features[0.r..rows, i].toList()))
        }
        val descriptor = mean + std + min + max
        val index = EmotionActivity.clf?.predict(descriptor)
        Log.e(EmotionActivity.TAG, index.toString())
        return labels!![index!!]
    }

    private fun calculateSD(numArray: List<Float>): Float {
        var sum = 0.0
        var standardDeviation = 0.0
        for (num in numArray) {
            sum += num
        }
        val mean = sum / numArray.size
        for (num in numArray) {
            standardDeviation += Math.pow(num - mean, 2.0)
        }
        val divider = numArray.size - 1
        return Math.sqrt(standardDeviation / divider).toFloat()
    }

    @ExperimentalTime
    fun recognizeLiveVideo(inTensorBuffer: FloatBuffer): String {
        val res = getFeatures(inTensorBuffer, Constants.COUNT_OF_FRAMES_PER_INFERENCE)
        return if (WelcomeActivity.isEngagement) {
            classifyFeaturesReg(res)
        } else {
            classifyFeaturesClass(res)
        }
    }

    private fun getFeatures(inTensorBuffer: FloatBuffer, numFrames: Int): FloatArray {
        val inputTensor = Tensor.fromBlob(
            inTensorBuffer, longArrayOf(
                numFrames.toLong(),
                3, //channels
                Constants.TARGET_FACE_SIZE.toLong(), Constants.TARGET_FACE_SIZE.toLong()
            )
        )
        val outputTensor: Tensor = module!!.forward(IValue.from(inputTensor)).toTensor()

        val scores = outputTensor.dataAsFloatArray
        Log.d(TAG, outputTensor.shape()[0].toString())
        Log.d(TAG, outputTensor.shape()[1].toString())

        return scores
    }

    @ExperimentalTime
    @WorkerThread
    fun recognizeVideo(fromMs: Int,
                       toMs: Int,
                       mmr: MediaMetadataRetriever): AnalysisResult? {
        var numFrames = 0
        var faces : MutableList<Bitmap> = mutableListOf()
        var bitmap: Bitmap? = null
        var resizedBitmap: Bitmap? = null
        var bbox: Rect?
        var lastBbox: Rect? = null

        for (i in 0 until Constants.COUNT_OF_FRAMES_PER_INFERENCE) {
            val timeUs = (1000 * (fromMs + ((toMs - fromMs) * i /
                    (Constants.COUNT_OF_FRAMES_PER_INFERENCE - 1.0)).toInt())).toLong()
            bitmap = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            resizedBitmap = WelcomeActivity.resize(bitmap, false)
            val start = System.nanoTime()
            val bboxes: Vector<Box> = WelcomeActivity.mtcnnFaceDetector!!.detectFaces(
                resizedBitmap!!,
                Constants.MIN_FACE_SIZE
            )
            val elapsed = (System.nanoTime() - start)/10000000
            Log.e(TAG, "Timecost to run MTCNN: $elapsed")

            val box: Box? = bboxes.maxByOrNull { box ->
                box.score
            }

            bbox = box?.transform2Rect()
            if ((EmotionActivity.videoDetector != null || EngagementActivity.videoDetector != null) &&  bbox != null) {
                val bboxOrig = Rect(
                    bitmap!!.width * bbox.left / resizedBitmap.width,
                    bitmap.height * bbox.top / resizedBitmap.height,
                    bitmap.width * bbox.right / resizedBitmap.width,
                    bitmap.height * bbox.bottom / resizedBitmap.height
                )
                faces.add(Bitmap.createScaledBitmap(Bitmap.createBitmap(bitmap,
                    bboxOrig.left,
                    bboxOrig.top,
                    bboxOrig.width(),
                    bboxOrig.height()),
                    Constants.TARGET_FACE_SIZE, Constants.TARGET_FACE_SIZE, false))

                lastBbox = bbox
                numFrames += 1
            }
        }

        if (numFrames > 0) {
            val inTensorBuffer = Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE*numFrames)

            for (i in 0 until numFrames) {
                val byteArrayOutputStream = ByteArrayOutputStream()
                faces[i]!!.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)

                TensorImageUtils.bitmapToFloatBuffer(
                    faces[i],
                    0,
                    0,
                    Constants.TARGET_FACE_SIZE,
                    Constants.TARGET_FACE_SIZE,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    inTensorBuffer,
                    (i * Constants.MODEL_INPUT_SIZE))
            }

            val features = getFeatures(inTensorBuffer, numFrames)
            var emotion = ""
            emotion = if (WelcomeActivity.isEngagement) {
                classifyFeaturesReg(features)
            } else {
                classifyFeaturesClass(features)
            }

            return AnalysisResult(
                Rect(
                    (bitmap!!.width * lastBbox!!.left / resizedBitmap!!.width),
                    bitmap.height * lastBbox.top / resizedBitmap.height,
                    (bitmap.width * lastBbox.right / resizedBitmap.width),
                    bitmap.height * lastBbox.bottom / resizedBitmap.height
                ), emotion, bitmap.width, bitmap.height
            )
        }
        return null
    }

    init {
        module = Module.load(assetFilePath(context, feat_len, filename))
        loadLabels(context)
    }
}
