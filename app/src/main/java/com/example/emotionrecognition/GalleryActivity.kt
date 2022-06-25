package com.example.emotionrecognition

import android.annotation.SuppressLint
import android.content.ContentResolver
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.annotation.WorkerThread
import androidx.appcompat.app.AppCompatActivity
import com.example.emotionrecognition.FeatureExtractor.Companion.applyToUiAnalyzeImageResult
import kotlinx.android.synthetic.main.activity_gallery.*
import kotlin.math.ceil


class GalleryActivity : Runnable, AppCompatActivity() {
    private var mThread: Thread? = Thread(this)
    private var mStopThread = true
    private var mLastAnalysisResultTime: Long = 0
    private var engagement = true

    @SuppressLint("WrongThread")
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gallery)

        val cR: ContentResolver = this.applicationContext.contentResolver
        engagement = this.baseContext is EngagementActivity
        val type = if (engagement) {
            cR.getType(EngagementActivity.content!!)
        }
        else {
            cR.getType(EmotionActivity.content!!)
        }
        if (type == "video/mp4") {
            val mp = MediaPlayer()
            if (engagement) {
                mp.setDataSource(applicationContext, EngagementActivity.content!!)
            }
            else {
                mp.setDataSource(applicationContext, EmotionActivity.content!!)
            }
            mp.prepare()
            val width = mp.videoWidth
            val height = mp.videoHeight
            if (width > height) {
                val layoutParams = video.layoutParams
                layoutParams.height = FrameLayout.LayoutParams.WRAP_CONTENT
                layoutParams.width = FrameLayout.LayoutParams.WRAP_CONTENT
            }
            video.visibility = View.VISIBLE
            startVideo()
        }
    }

    private fun startVideo() {
        if (engagement) {
            video?.setVideoURI(EngagementActivity.content)
        }
        else {
            video?.setVideoURI(EmotionActivity.content)
        }
        video?.setOnCompletionListener {
            TopPanel.visibility = View.VISIBLE
            Back.visibility = View.VISIBLE
        }
        video.setOnPreparedListener { mp: MediaPlayer ->
            mp.setScreenOnWhilePlaying(false)
        }
        video?.start()
        if (mThread != null && mThread!!.isAlive) {
            try {
                mThread!!.join()
            } catch (e: InterruptedException) {
                Log.e("GalleryActivity", e.localizedMessage)
            }
        }
        mStopThread = false
        mThread?.start()
    }

    @OptIn(kotlin.time.ExperimentalTime::class)
    @WorkerThread
    override fun run() {   // video recognition
        val mmr = MediaMetadataRetriever()
        if (engagement) {
            mmr.setDataSource(this.applicationContext, EngagementActivity.content)
        }
        else {
            mmr.setDataSource(this.applicationContext, EmotionActivity.content)
        }
        val stringDuration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
        val durationMs = stringDuration!!.toDouble()

        // for each second of the video, make inference to get the class label
        val durationTo = durationMs.toInt()
        var from = 0

        while (!mStopThread && from < durationTo) {
            from += 1000
            var to = from + 1000
            if (to > durationTo) to = ceil(durationMs).toInt()

            val start = System.nanoTime()
            val result = if (engagement) {
                EngagementActivity.videoDetector!!.recognizeVideo(from, to, mmr)
            }
            else {
                EmotionActivity.videoDetector!!.recognizeVideo(from, to, mmr)
            }

            val elapsed = (System.nanoTime() - start)/10000000
            Log.e("GalleryActivity", String.format("Timecost to run PyTorch model inference: $elapsed"))

            from += elapsed.toInt()

            if (from > video.currentPosition) {
                try {
                    Thread.sleep((from - video!!.currentPosition).toLong())
                } catch (e: InterruptedException) {
                    Log.e("GalleryActivity", "Thread sleep exception: " + e.localizedMessage)
                }
            }
            while (!video!!.isPlaying) {
                if (mStopThread || video!!.currentPosition >= video!!.duration) break
                try {
                    Thread.sleep(100)
                } catch (e: InterruptedException) {
                    Log.e("GalleryActivity", "Thread sleep exception: " + e.localizedMessage)
                }
            }

            if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime > 500) {
                runOnUiThread {
                    val overlay: ImageView = findViewById(R.id.gallery_overlay)
                    overlay.setImageResource(android.R.color.transparent)
                    val text: TextView = findViewById(R.id.galleryText)
                    text.visibility = View.VISIBLE
                    text.text = "NO FACES DETECTED"
                }
            }

            if (result != null){
                runOnUiThread {
                    mLastAnalysisResultTime = SystemClock.elapsedRealtime()
                    val text: TextView = findViewById(R.id.galleryText)
                    text.visibility = View.GONE
                    applyToUiAnalyzeImageResult(result, result!!.width, result.height, findViewById(R.id.gallery_overlay))
                }
            }
        }
        mmr.release()
    }

    private fun stopVideo() {
        video.stopPlayback()
        mStopThread = true
    }

    override fun onPause() {
        super.onPause()
        video.pause()
    }

    override fun onStop() {
        super.onStop()
        stopVideo()
    }

    fun back(view: View) {
        finish()
    }
}