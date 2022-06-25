package com.example.emotionrecognition

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity

class EmotionActivity : AppCompatActivity() {
    companion object {
        var content: Uri? = null
        const val TAG = "EmotionActivity"
        var videoDetector: FeatureExtractor? = null
        var clf: LinearSVC? = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_emotion)
        init()
    }

    private fun init() {
        try {
            videoDetector = FeatureExtractor(applicationContext, 1024, "EfficientNet_B0.pt")
        } catch (e: java.lang.Exception) {
            Log.e(TAG, "Exception initializing feature extractor!", e)
        }

        try {
            clf = LinearSVC(applicationContext)
        } catch (e: java.lang.Exception) {
            Log.e(TAG, "Exception initializing classifier!", e)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == Constants.REQUEST_ACCESS_TYPE) {
            content = data?.data
            val intentNextStep = Intent(this, GalleryActivity::class.java)
            startActivity(intentNextStep)
        }
    }

    fun startGallery(view: View) {
        val intent = Intent()
        intent.type = "video/*"
        intent.action = Intent.ACTION_PICK
        startActivityForResult(intent, Constants.REQUEST_ACCESS_TYPE)
    }

    fun startLive(view: View) {
        val intent = Intent(this@EmotionActivity, CameraActivity::class.java)
        startActivity(intent)
    }

    fun back(view: View) {
        finish()
    }
}