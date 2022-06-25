package com.example.emotionrecognition

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import java.util.*


class LinearSVC(context: Context) {
    internal class Classifier {
        val coefficients: Array<DoubleArray> = arrayOf()
        val intercepts: DoubleArray = doubleArrayOf()
    }

    private val clf: Classifier

    fun predict(features: List<Float>): Int {
        var classIdx = 0
        var classVal = Double.NEGATIVE_INFINITY
        var i = 0
        val il = clf.intercepts.size
        while (i < il) {
            var prob = 0.0
            var j = 0
            val jl: Int = clf.coefficients[0].size
            while (j < jl) {
                prob += clf.coefficients[i][j] * features[j]
                j++
            }
            if (prob + clf.intercepts[i] > classVal) {
                classVal = prob + clf.intercepts[i]
                classIdx = i
            }
            i++
        }
        return classIdx
    }

    init {
        val jsonStr = Scanner(LinearSVR.getFileFromAsset(context, 1024, "classifier.json")).useDelimiter("\\Z").next()
        Log.e(EngagementActivity.TAG, jsonStr)
        clf = Gson().fromJson(jsonStr, Classifier::class.java)
    }
}