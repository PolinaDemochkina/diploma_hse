package com.example.emotionrecognition

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.emotionrecognition.mtcnn.FaceDetector
import java.util.HashMap

class WelcomeActivity : AppCompatActivity() {
    companion object {
        var content: Uri? = null
        const val TAG = "WelcomeActivity"
        var mtcnnFaceDetector: FaceDetector? = null

        fun resize(frame: Bitmap?, image: Boolean): Bitmap? {
            val ratio: Float = if (image) {
                Math.min(frame!!.width, frame.height) / Constants.TARGET_IMAGE_SIZE.toFloat()
            } else {
                Math.min(frame!!.width, frame.height) / Constants.TARGET_VIDEO_SIZE.toFloat()
            }

            return Bitmap.createScaledBitmap(
                frame,
                (frame.width / ratio).toInt(),
                (frame.height / ratio).toInt(),
                false
            )
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)
        if (!allPermissionsGranted()) {
            getRequiredPermissions()?.let {
                ActivityCompat.requestPermissions(
                    this,
                    it,
                    Constants.REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS
                )
            }
        }
        else
            init()
    }

    private fun getRequiredPermissions(): Array<String?>? {
        return try {
            val info = packageManager
                .getPackageInfo(packageName, PackageManager.GET_PERMISSIONS)
            val ps = info.requestedPermissions
            if (ps != null && ps.isNotEmpty()) {
                ps
            } else {
                arrayOfNulls(0)
            }
        } catch (e: java.lang.Exception) {
            arrayOfNulls(0)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            Constants.REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS -> {
                val perms: MutableMap<String, Int> = HashMap()
                var allGranted = true
                var i = 0
                while (i < permissions.size) {
                    perms[permissions[i]] = grantResults[i]
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED) allGranted = false
                    i++
                }

                if (allGranted) {
                    // All Permissions Granted
                    init()
                } else {
                    // Permission Denied
                    Toast.makeText(
                        this@WelcomeActivity,
                        "Some Permission is Denied",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                    finish()
                }
            }
            else -> super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    private fun allPermissionsGranted(): Boolean {
        for (permission in getRequiredPermissions()!!) {
            permission?.let { ContextCompat.checkSelfPermission(this, it) }
            if (permission?.let { ContextCompat.checkSelfPermission(this, it) }
                != PackageManager.PERMISSION_GRANTED
            ) {
                return false

            }
        }
        return true
    }

    private fun init() {
        try {
            mtcnnFaceDetector = FaceDetector.create(assets)
        } catch (e: Exception) {
            Log.e(TAG, "Exception initializing MTCNNModel!${e.stackTraceToString()}")
        }
    }

    fun startEmotion(view: View) {
        val intent = Intent(this@WelcomeActivity, EmotionActivity::class.java)
        startActivity(intent)
    }

    fun startEngagement(view: View) {
        val intent = Intent(this@WelcomeActivity, EngagementActivity::class.java)
        startActivity(intent)
    }
}