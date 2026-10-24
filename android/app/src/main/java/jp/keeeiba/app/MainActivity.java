package jp.keeeiba.app;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.CookieManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * 2強レース(個人用)。予想アプリ(GitHub Pages)を1画面で出す。
 * ブラウザの予想アプリからは netkeiba を読めないので、オッズをその場で取るときはアプリが代わりに取る(KeibaApp.fetch)。
 */
public class MainActivity extends Activity {
    static final String APP_URL = "https://honda1986.github.io/keeeiba-app/keiba/?app=1";

    WebView web;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        CookieManager.getInstance().setAcceptCookie(true);
        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);          // 予想アプリの localStorage(買い目の記録など)
        s.setDatabaseEnabled(true);
        s.setSupportMultipleWindows(false);
        web.setWebChromeClient(new WebChromeClient());
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                Uri u = req.getUrl();
                if ("honda1986.github.io".equals(u.getHost())) return false;
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, u));     // ほかのサイトはブラウザで開く
                } catch (Exception e) {
                    toast("開けません: " + u);
                }
                return true;
            }
        });
        web.addJavascriptInterface(new Bridge(), "KeibaApp");
        setContentView(web);
        if (saved != null) web.restoreState(saved);
        if (web.getUrl() == null) web.loadUrl(APP_URL);
    }

    void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }

    /** 予想アプリ(Web)から呼ばれる */
    class Bridge {
        /** netkeiba(race.netkeiba.com だけ)のページを取って、文字を予想アプリに返す(window.kbFetched(id, 文字, エラー)) */
        @JavascriptInterface
        public void fetch(String url, String id) {
            if (url == null || !url.startsWith("https://race.netkeiba.com/")) {
                fetched(id, null, "このアドレスは取れません");
                return;
            }
            new Thread(() -> {
                String body = null, err = null;
                try {
                    HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
                    c.setConnectTimeout(10000);
                    c.setReadTimeout(15000);
                    c.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36");
                    int code = c.getResponseCode();
                    if (code >= 400) throw new Exception("HTTP " + code);
                    ByteArrayOutputStream b = new ByteArrayOutputStream();
                    try (InputStream in = c.getInputStream()) {
                        byte[] buf = new byte[8192];
                        int n;
                        while ((n = in.read(buf)) > 0) b.write(buf, 0, n);
                    }
                    // 文字コード: 返事に書いてあればそれ、無ければ API は UTF-8、ページは EUC-JP(netkeiba)
                    String ct = c.getContentType() == null ? "" : c.getContentType();
                    String cs = ct.toLowerCase().contains("charset=") ? ct.substring(ct.toLowerCase().indexOf("charset=") + 8).trim()
                            : url.contains("/api/") ? "UTF-8" : "EUC-JP";
                    body = b.toString(cs.split(";")[0].replace("\"", "").trim());
                } catch (Exception e) {
                    err = e.toString();
                }
                fetched(id, body, err);
            }).start();
        }

        void fetched(String id, String body, String err) {
            String js = "window.kbFetched&&window.kbFetched(" + JSONObject.quote(id) + "," + (body == null ? "null" : JSONObject.quote(body))
                    + "," + JSONObject.quote(err == null ? "" : err) + ")";
            runOnUiThread(() -> web.evaluateJavascript(js, null));
        }

        @JavascriptInterface
        public String version() {
            try {
                return getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
            } catch (Exception e) {
                return "";
            }
        }
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }

    @Override
    protected void onPause() {
        super.onPause();
        CookieManager.getInstance().flush();
    }
}
