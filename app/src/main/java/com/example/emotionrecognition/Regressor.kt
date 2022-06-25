package com.example.emotionrecognition

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.gson.Gson
import java.io.*
import java.util.*


class LinearSVR(context: Context) {
    internal class Regressor {
        val coefficients: DoubleArray = doubleArrayOf()
        val intercepts: DoubleArray = doubleArrayOf()
    }

    private val clf: Regressor

    companion object {
        @Throws(IOException::class)
        fun getFileFromAsset(context: Context, featureSize: Int, filename: String): File? {
            val assetManager: AssetManager = context.assets
            var inputStream: InputStream? = null
            var outputStream: OutputStream? = null
            var file: File? = null
            val buffer = ByteArray(4 * featureSize)

            try {
                inputStream = assetManager.open(filename)
                file = File(context.cacheDir, filename)
                outputStream = FileOutputStream(file)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            } catch (e: Exception) {
                e.printStackTrace() // handle exception, define IOException and others
            } finally {
                try {
                    outputStream?.close()
                    inputStream?.close()
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
            return file
        }
    }

    fun predict(features: List<Float>): Double {
        var prob = 0.0
        var j = 0
        val jl: Int = clf.coefficients.size
        while (j < jl) {
            prob += clf.coefficients[j] * features[j]
            j++
        }
        var classVal = prob + clf.intercepts[0]

        return classVal
    }

    init {
        val jsonStr = Scanner(getFileFromAsset(context, 1280, "regressor.json")).useDelimiter("\\Z").next()
        Log.e(EngagementActivity.TAG, jsonStr)
        clf = Gson().fromJson(jsonStr, Regressor::class.java)
    }
}