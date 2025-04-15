import yfinance as yf

if __name__ == '__main__':
    # دانلود داده‌های تاریخی بیت‌کوین از 2014 تا 2024
    data = yf.download('BTC-USD', start='2014-01-01', end='2024-12-31')

    # ذخیره داده‌ها در فایل CSV
    data.to_csv('btc_usd_history.csv')
