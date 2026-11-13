package org.cvision.assist;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.webkit.ConsoleMessage;
import android.webkit.CookieManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.ServiceWorkerClient;
import android.webkit.ServiceWorkerController;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** An offline, single-origin shell. All inference stays inside the WebView. */
public final class MainActivity extends Activity {
    private static final String START_URL = "https://appassets.androidplatform.net/assist?threads=1";
    private static final int MEDIA_PERMISSIONS = 41;
    private FrameLayout container;
    private WebView web;
    private PermissionRequest pendingPermission;
    private boolean permissionDialog;
    private boolean resumed;
    private volatile boolean destroyed;
    private boolean backPending;
    private TextToSpeech tts;
    private ServiceWorkerController workerController;
    private volatile boolean speechAvailable;
    private volatile int speechGeneration;

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        container = new FrameLayout(this);
        container.setBackgroundColor(Color.rgb(5, 7, 12));
        setContentView(container);
        if (Build.VERSION.SDK_INT >= 30) {
            getWindow().setDecorFitsSystemWindows(false);
            container.setOnApplyWindowInsetsListener((view, insets) -> {
                android.graphics.Insets safe = insets.getInsets(
                        WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
                return insets;
            });
        }
        if (Build.VERSION.SDK_INT >= 33) {
            getOnBackInvokedDispatcher().registerOnBackInvokedCallback(
                    android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, this::handleBack);
        }
        // Post the callback so tts has been assigned even with a very fast engine.
        tts = new TextToSpeech(getApplicationContext(), status ->
                container.post(() -> configureVoice(status)));
    }

    @Override protected void onResume() {
        super.onResume();
        resumed = true;
        hideSystemBars();
        if (web == null) createWebView();
        if (web != null) web.onResume();
        if (!permissionDialog) completePermission();
    }

    @Override protected void onPause() {
        resumed = false;
        stopSpeech();
        // A runtime permission dialog may pause the Activity. Keep its own
        // request alive until the result; backgrounding still tears down onStop.
        if (!permissionDialog) {
            cancelPermission();
            WebView current = web;
            if (current != null) {
                current.evaluateJavascript("window.assist && window.assist.stop()", ignored -> {
                    if (!resumed && web == current) current.onPause();
                });
            }
        }
        super.onPause();
    }

    @Override protected void onStop() {
        // Destroying guarantees capture/worker release even if JS is stalled.
        // Returning recreates the start screen; localStorage keeps preferences.
        releaseWebView();
        super.onStop();
    }

    @Override protected void onDestroy() {
        destroyed = true;
        releaseWebView();
        if (tts != null) { tts.shutdown(); tts = null; }
        super.onDestroy();
    }

    @Override public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) hideSystemBars();
    }

    @SuppressWarnings("deprecation")
    private void hideSystemBars() {
        if (Build.VERSION.SDK_INT >= 30) {
            WindowInsetsController controls = getWindow().getInsetsController();
            if (controls != null) {
                controls.hide(WindowInsets.Type.systemBars());
                controls.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            }
        } else {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION);
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    @SuppressWarnings("deprecation")
    private void createWebView() {
        if (destroyed) return;
        try {
            PackageInfo engine = WebView.getCurrentWebViewPackage();
            if (engine == null) {
                showError("Android System WebView is missing. Install or enable a compatible WebView, then retry.");
                return;
            }
            int major = Integer.parseInt(engine.versionName.split("\\.")[0]);
            if (major < 91) {
                showError("This WebView is too old for the vision models. Update Android System WebView (91 or newer) and retry.");
                return;
            }
            WebView view = new WebView(this);
            web = view;
            view.setBackgroundColor(Color.rgb(5, 7, 12));
            WebSettings settings = view.getSettings();
            settings.setJavaScriptEnabled(true);
            settings.setDomStorageEnabled(true);
            settings.setAllowFileAccess(false);
            settings.setAllowContentAccess(false);
            settings.setAllowFileAccessFromFileURLs(false);
            settings.setAllowUniversalAccessFromFileURLs(false);
            settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
            settings.setMediaPlaybackRequiresUserGesture(false);
            settings.setJavaScriptCanOpenWindowsAutomatically(false);
            settings.setSupportMultipleWindows(false);
            settings.setGeolocationEnabled(false);
            settings.setCacheMode(WebSettings.LOAD_NO_CACHE);
            CookieManager.getInstance().setAcceptCookie(false);
            CookieManager.getInstance().setAcceptThirdPartyCookies(view, false);
            view.addJavascriptInterface(new SpeechBridge(), "AssistNative");
            view.setWebViewClient(new LocalClient());
            view.setWebChromeClient(new WebChromeClient() {
                @Override public void onPermissionRequest(PermissionRequest request) {
                    requestMediaPermission(request);
                }
                @Override public void onPermissionRequestCanceled(PermissionRequest request) {
                    if (pendingPermission == request) pendingPermission = null;
                }
                @Override public boolean onConsoleMessage(ConsoleMessage message) {
                    // Do not log recognised label text or captions to logcat.
                    return true;
                }
            });
            ServiceWorkerController workers = ServiceWorkerController.getInstance();
            workerController = workers;
            workers.getServiceWorkerWebSettings().setAllowContentAccess(false);
            workers.getServiceWorkerWebSettings().setAllowFileAccess(false);
            workers.setServiceWorkerClient(new ServiceWorkerClient() {
                @Override public WebResourceResponse shouldInterceptRequest(WebResourceRequest request) {
                    return intercept(request);
                }
            });
            container.removeAllViews();
            container.addView(view, new FrameLayout.LayoutParams(-1, -1));
            view.requestFocus();
            view.loadUrl(START_URL);
        } catch (RuntimeException error) {
            releaseWebView();
            showError("The Android web engine could not start. Update or enable Android System WebView, then retry.");
        }
    }

    private final class LocalClient extends WebViewClient {
        @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
            return intercept(request);
        }

        @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
            // External links never enter the bridge-enabled WebView.
            return !request.isForMainFrame() || !isAppPage(request.getUrl().toString());
        }

        @Override public void onPageFinished(WebView view, String url) {
            if (web != view || !isAppPage(url)) return;
            notifySpeechAvailability();
        }

        @Override public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
            if (request.isForMainFrame() && web == view) {
                container.post(() -> {
                    if (web == view) {
                        releaseWebView();
                        showError("The bundled app could not open. Retry, or reinstall the APK if the problem continues.");
                    }
                });
            }
        }

        @Override public boolean onRenderProcessGone(WebView view, RenderProcessGoneDetail detail) {
            if (web == view) {
                releaseWebView();
                showError("The web engine stopped, possibly because memory was low. Close other apps and retry.");
            } else view.destroy();
            return true;
        }
    }

    private static boolean isAppPage(String url) {
        return "web/assist.html".equals(BundlePolicy.assetPath(url));
    }

    private WebResourceResponse intercept(WebResourceRequest request) {
        String url = request.getUrl().toString();
        if (!"GET".equals(request.getMethod())) return response(405, "Method Not Allowed", "text/plain", null);
        String path = BundlePolicy.assetPath(url);
        if (path != null) {
            try { return response(200, "OK", mime(path), getAssets().open(path)); }
            catch (IOException missing) { return response(404, "Not Found", "text/plain", null); }
        }
        // The only network endpoint is the optional barcode lookup. The page
        // chooses whether to call it; media never uses a network endpoint.
        if (!request.isForMainFrame() && BundlePolicy.isAllowedNetwork(url)) return null;
        return response(403, "Forbidden", "text/plain", null);
    }

    private static String mime(String path) {
        if (path.endsWith(".html")) return "text/html";
        if (path.endsWith(".js") || path.endsWith(".mjs")) return "text/javascript";
        if (path.endsWith(".wasm")) return "application/wasm";
        if (path.endsWith(".json")) return "application/json";
        if (path.endsWith(".txt")) return "text/plain";
        if (path.endsWith(".css")) return "text/css";
        if (path.endsWith(".svg")) return "image/svg+xml";
        if (path.endsWith(".png")) return "image/png";
        return "application/octet-stream";
    }

    private static WebResourceResponse response(int status, String reason, String mime, InputStream stream) {
        if (stream == null) stream = new ByteArrayInputStream(reason.getBytes(StandardCharsets.UTF_8));
        Map<String, String> headers = new HashMap<>();
        headers.put("Cross-Origin-Opener-Policy", "same-origin");
        headers.put("Cross-Origin-Embedder-Policy", "require-corp");
        headers.put("Cross-Origin-Resource-Policy", "same-origin");
        headers.put("X-Content-Type-Options", "nosniff");
        headers.put("Cache-Control", "no-store");
        headers.put("Referrer-Policy", "no-referrer");
        headers.put("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-eval'; "
                + "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; "
                + "worker-src 'self' blob:; connect-src 'self' https://world.openfoodfacts.org; "
                + "frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'");
        return new WebResourceResponse(mime, mime.startsWith("text/") || mime.equals("application/json") ? "UTF-8" : null,
                status, reason, headers, stream);
    }

    private void requestMediaPermission(PermissionRequest request) {
        if (!resumed || web == null || !isAppPage(web.getUrl())
                || !BundlePolicy.isLocal(request.getOrigin().toString()) || pendingPermission != null || permissionDialog) {
            request.deny();
            return;
        }
        List<String> permissions = new ArrayList<>();
        for (String resource : request.getResources()) {
            String permission = permissionFor(resource);
            if (permission == null) { request.deny(); return; }
            if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED && !permissions.contains(permission)) {
                permissions.add(permission);
            }
        }
        pendingPermission = request;
        if (permissions.isEmpty()) completePermission();
        else {
            permissionDialog = true;
            requestPermissions(permissions.toArray(new String[0]), MEDIA_PERMISSIONS);
        }
    }

    private static String permissionFor(String resource) {
        if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(resource)) return Manifest.permission.CAMERA;
        if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(resource)) return Manifest.permission.RECORD_AUDIO;
        return null;
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MEDIA_PERMISSIONS) {
            permissionDialog = false;
            completePermission();
        }
    }

    private void completePermission() {
        if (pendingPermission == null || permissionDialog || !resumed) return;
        PermissionRequest request = pendingPermission;
        pendingPermission = null;
        if (web == null || !isAppPage(web.getUrl()) || !BundlePolicy.isLocal(request.getOrigin().toString())) {
            request.deny(); return;
        }
        List<String> granted = new ArrayList<>();
        for (String resource : request.getResources()) {
            String permission = permissionFor(resource);
            if (permission == null || checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
                request.deny();
                Toast.makeText(this, "Camera or microphone permission was denied. You can change it in Android app settings.", Toast.LENGTH_LONG).show();
                return;
            }
            granted.add(resource);
        }
        if (granted.isEmpty()) request.deny();
        else request.grant(granted.toArray(new String[0]));
    }

    private void cancelPermission() {
        if (pendingPermission != null) { pendingPermission.deny(); pendingPermission = null; }
    }

    @SuppressWarnings("deprecation")
    // API 26-32 fallback. API 33+ uses the OnBackInvokedDispatcher registered
    // in onCreate, including Android 16's gesture path.
    @SuppressLint("GestureBackNavigation")
    @Override public void onBackPressed() { handleBack(); }

    private void handleBack() {
        if (backPending) return;
        WebView current = web;
        if (current == null) { finish(); return; }
        backPending = true;
        current.evaluateJavascript("(() => { const a = window.assist; if (!a) return false; "
                + "if (a.starting) { a.stop(); return true; } if (a.running) { a.back(); return true; } return false; })()",
                handled -> {
                    backPending = false;
                    if (web == current && !"true".equals(handled)) finish();
                });
    }

    private void releaseWebView() {
        cancelPermission();
        stopSpeech();
        backPending = false;
        if (workerController != null) {
            workerController.setServiceWorkerClient(null);
            workerController = null;
        }
        WebView current = web;
        web = null;
        if (current != null) {
            container.removeView(current);
            current.removeJavascriptInterface("AssistNative");
            current.stopLoading();
            current.destroy();
        }
    }

    private void showError(String explanation) {
        if (destroyed) return;
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (18 * getResources().getDisplayMetrics().density);
        panel.setPadding(pad, pad, pad, pad);
        TextView title = new TextView(this);
        title.setText(R.string.app_name); title.setTextSize(22); title.setTextColor(Color.WHITE);
        TextView body = new TextView(this);
        body.setText(explanation); body.setTextSize(16); body.setTextColor(Color.LTGRAY);
        panel.addView(title); panel.addView(body);
        Button retry = new Button(this);
        retry.setText(R.string.retry); retry.setOnClickListener(view -> createWebView());
        panel.addView(retry);
        Button settings = new Button(this);
        settings.setText(R.string.app_settings);
        settings.setOnClickListener(view -> startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.parse("package:" + getPackageName()))));
        panel.addView(settings);
        container.removeAllViews(); container.addView(panel);
        retry.requestFocus();
    }

    private void configureVoice(int status) {
        if (destroyed || tts == null) return;
        speechAvailable = false;
        if (status == TextToSpeech.SUCCESS) {
            Set<Voice> installed = tts.getVoices();
            if (installed != null) {
                List<Voice> candidates = new ArrayList<>();
                for (Voice voice : installed) {
                    if (!voice.isNetworkConnectionRequired() && "en".equals(voice.getLocale().getLanguage())
                            && (voice.getFeatures() == null || !voice.getFeatures().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED))) {
                        candidates.add(voice);
                    }
                }
                candidates.sort(Comparator.comparing(Voice::getName));
                if (!candidates.isEmpty()) speechAvailable = tts.setVoice(candidates.get(0)) == TextToSpeech.SUCCESS;
            }
        }
        notifySpeechAvailability();
    }

    private void notifySpeechAvailability() {
        if (web != null && isAppPage(web.getUrl())) {
            web.evaluateJavascript("window.dispatchEvent(new Event('assist-native-speech'))", null);
        }
    }

    private void stopSpeech() {
        speechGeneration++;
        if (tts != null) tts.stop();
    }

    public final class SpeechBridge {
        @JavascriptInterface public boolean isSpeechAvailable() { return speechAvailable && !destroyed; }

        @JavascriptInterface public void stopSpeaking() { runOnUiThread(MainActivity.this::stopSpeech); }

        @JavascriptInterface public void speak(String text) {
            final int generation = speechGeneration;
            runOnUiThread(() -> {
                if (generation != speechGeneration || !resumed || !speechAvailable || tts == null
                        || web == null || !isAppPage(web.getUrl()) || text == null || text.trim().isEmpty()) return;
                if (text.length() > 32768) {
                    Toast.makeText(MainActivity.this, "The text is too long to read aloud in one pass.", Toast.LENGTH_LONG).show();
                    return;
                }
                tts.stop();
                int limit = Math.min(3500, TextToSpeech.getMaxSpeechInputLength());
                for (int offset = 0, part = 0; offset < text.length(); part++) {
                    int end = Math.min(text.length(), offset + limit);
                    if (end < text.length()) {
                        int space = text.lastIndexOf(' ', end);
                        if (space > offset) end = space;
                    }
                    int result = tts.speak(text.substring(offset, end), TextToSpeech.QUEUE_ADD, null, "assist-" + generation + "-" + part);
                    if (result == TextToSpeech.ERROR) {
                        tts.stop(); speechAvailable = false; notifySpeechAvailability(); break;
                    }
                    offset = end;
                }
            });
        }
    }
}
