package com.example.emotionrecognition

import android.content.Intent
import android.content.pm.PackageManager
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
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)
    }

    fun startMain(view: View) {
        val intent = Intent(this@WelcomeActivity, MainActivity::class.java)
        startActivity(intent)
    }
}