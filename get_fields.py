def getfields(windows,rolling_windows):
    fields = []
    names = []
    

        



    fields += [
        "(($close-$open)/$open)*100",
        "(($high-$low)/$open)*100",
        "($close-$open)/($high-$low+1e-12)",
        "($high-Greater($open, $close))/$open",
        "($high-Greater($open, $close))/($high-$low+1e-12)",
        "(Less($open, $close)-$low)/$open",
        "(Less($open, $close)-$low)/($high-$low+1e-12)",
        "(2*$close-$high-$low)/$open",
        "(2*$close-$high-$low)/($high-$low+1e-12)",
    ]
    names += [
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
    ]
    feature = ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
    for field in feature:
        field = field.lower()
        fields += [
            "Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field
            for d in windows
        ]
        names += [field.upper() + str(d) for d in windows]
    fields += [
        "Log($volume/(Ref($volume, %d)+1e-12)+1)" % d
        for d in windows
    ]
    names += ["VOLUME" + str(d) for d in windows]
    fields += [
        "((Ref($open,%d) + Ref($close,%d))/2)/Ref($close, %d)" % (d + 1, d + 1, d)
        if d != 0
        else "((Ref($open,%d) + Ref($close,%d))/2)/$close" % (d + 1, d + 1)
        for d in windows
    ]
    names += ["HEIASHO" + str(d) for d in windows]
    fields += [
        "((Ref($open,%d) + Ref($close,%d))/2)/$close" % (d + 1, d + 1) for d in windows
    ]
    names += ["HEIASHON" + str(d) for d in windows]
    fields += [
        "((Ref($adj, %d)*3/4) + (Ref($close, %d)*1/4))/Ref($close, %d)"
        % (d, d, d)
        if d != 0
        else "(($adj*3/4) + ($close*1/4))/$close"
        for d in windows
    ]
    names += ["HEIASHC" + str(d) for d in windows]
    fields += [
        "((Ref($adj, %d)*3/4) + (Ref($close, %d)*1/4))/$close"
        % (d, d)
        if d != 0
        else "(($adj*3/4) + ($close*1/4))/$close"
        for d in windows
    ]
    names += ["HEIASHCN" + str(d) for d in windows]
    windows = rolling_windows
    # fields += ["SlopeBTC($close, %d)/RefBTC($close,0)" % d for d in windows]
    # names += ["SLPBTC%d" % d for d in windows]
    # fields += ["SlopeETH($close, %d)/RefETH($close,0)" % d for d in windows]
    # names += ["SLPETH%d" % d for d in windows]
    # fields += ["DeltaBTC($close, %d)/RefBTC($close,0)" % d for d in windows]
    # names += ["DELBTC%d" % d for d in windows]
    # fields += ["DeltaETH($close, %d)/RefETH($close,0)" % d for d in windows]
    # names += ["DELETH%d" % d for d in windows]
    
    
    # fields += ["Corr(RefBTC($close,0)/RefBTC($close,1), Log($volume/(Ref($volume, 1)+1e-12)+1), %d)" % (d) for d in windows]
    # names += ["CORRBTCV%d" % d for d in windows]
    # fields += ["Corr(RefETH($close,0)/RefETH($close,1), Log($volume/(Ref($volume, 1)+1e-12)+1), %d)" % (d) for d in windows]
    # names += ["CORRETHV%d" % d for d in windows]
    # fields += ["Corr(RefBTC($close,0)/RefBTC($close,1), $close/Ref($close, 1), %d)" % (d) for d in windows]
    # names += ["CORRBTC%d" % d for d in windows]
    # fields += ["Corr(RefETH($close,0)/RefETH($close,1), $close/Ref($close, 1), %d)" % (d) for d in windows]
    # names += ["CORRETH%d" % d for d in windows]
    
    # fields += ["Corr(Log(RefBTC($volume,0)/RefBTC($volume, 1)+1), Log($volume/(Ref($volume, 1)+1e-12)+1), %d)" % (d) for d in windows]
    # names += ["CORRBTCVV%d" % d for d in windows]
    # fields += ["Corr(Log(RefETH($volume,0)/RefETH($volume, 1)+1), Log($volume/(Ref($volume, 1)+1e-12)+1), %d)" % (d) for d in windows]
    # names += ["CORRETHVV%d" % d for d in windows]
    
    fields += ["(Max($high, %d)/$high)" % d for d in windows]
    names += ["MAXGE%d" % d for d in windows]
    fields += ["(Min($low, %d)/$low)" % d for d in windows]
    names += ["MINGE%d" % d for d in windows]
    fields += ["(Max($high, %d)/$close)" % d for d in windows]
    names += ["MAXCGE%d" % d for d in windows]
    fields += ["(Min($low, %d)/$close)" % d for d in windows]
    names += ["MINCGE%d" % d for d in windows]
    # STOCHASTIC STUFF
    fields += [
        "(($close - Min($low,%d)+1e-6)/(Max($high,%d) - Min($low,%d)+1e-6))" % (d, d, d)
        for d in windows
    ]
    names += ["STOCH%d" % d for d in windows]
    fields += [
        "Mean((($close - Min($low,%d)+1e-6)/(Max($high,%d) - Min($low,%d)+1e-6)),%d)"
        % (d, d, d, d)
        for d in windows
    ]
    names += ["STOCHEMA%d" % d for d in windows]
    fields += [
        "Mean(100*(($close - Min($low,%d)+1e-6)/(Max($high,%d) - Min($low,%d)+1e-6)),%d) - 100*(($close - Min($low,%d)+1e-6)/(Max($high,%d) - Min($low,%d)+1e-6))"
        % (d, d, d, d, d, d, d)
        for d in windows
    ]
    names += ["STOCHEMAD%d" % d for d in windows]
    # #EMA/MACD
    fields += ["(EMA($close,12) - EMA($close,26))/$close"]
    names += ["MACDs"]
    fields += ["EMA((EMA($close,12) - EMA($close,26)),9)/$close"]
    names += ["MACDF"]
    fields += [
        "(EMA((EMA($close,12) - EMA($close,26)),9) - EMA($close,12) - EMA($close,26))/$close"
    ]
    names += ["MACDI"]
    
    fields += ["(EMA($close,%d) - EMA($close,%d//0.461538))/$close" % (d, d) for d in windows] 
    names += ["MACDs%d" % d for d in windows]
    fields += ["EMA((EMA($close,%d) - EMA($close,%d//0.461538)),%d//1.3333)/$close" % (d, d, d) for d in windows]
    names += ["MACDFDAY%d" % d for d in windows]
    fields += [
        "(EMA((EMA($close,%d) - EMA($close,%d//0.461538)),%d//1.3333) - EMA($close,%d) - EMA($close,%d//0.461538))/$close" % (d, d, d, d, d) for d in windows
    ]
    names += ["MACDIDAY%d" % d for d in windows]
    # #BOLLINGER
    fields += [
        "(Mean($adj,%d) + 2*Std($adj,%d))/$close"
        % (d, d)
        for d in windows
    ]
    names += ["BOLU%d" % d for d in windows]
    fields += [
        "(Mean($adj,%d) - 2*Std($adj,%d))/$close"
        % (d, d)
        for d in windows
    ]
    names += ["BOLD%d" % d for d in windows]
    # RSI
    fields += [
        "100 - (100/(1 + (Sum(If($close - Ref($close,1)>0,$close - Ref($close,1),0),%d)+1e-6)/(Sum(If($close - Ref($close,1)<0,$close - Ref($close,1),0),%d)+1e-6)))"
        % (d, d)
        for d in windows
    ]
    names += ["RSI%d" % d for d in windows]
    
    # fields += ["Ref($close, %d)/$close" % d for d in windows]
    # names += ["ROC%d" % d for d in windows]
    # fields += ["Mean($close, %d)/$close" % d for d in windows]
    # names += ["MA%d" % d for d in windows]
    # fields += ["Std($close, %d)/$close" % d for d in windows]
    # names += ["STD%d" % d for d in windows]
    # fields += ["Slope($close, %d)/$close" % d for d in windows]
    # names += ["BETA%d" % d for d in windows]
    # fields += ["Rsquare($close, %d)" % d for d in windows]
    # names += ["RSQR%d" % d for d in windows]
    # fields += ["Resi($close, %d)/$close" % d for d in windows]
    # names += ["RESI%d" % d for d in windows]
    # fields += ["Max($high, %d)/$close" % d for d in windows]
    # names += ["MAX%d" % d for d in windows]
    # fields += ["Min($low, %d)/$close" % d for d in windows]
    # names += ["MIN%d" % d for d in windows]
    # fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
    # names += ["QTLU%d" % d for d in windows]
    # fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
    # names += ["QTLD%d" % d for d in windows]
    # # fields += ["Rank($close, %d)" % d for d in windows]
    # # names += ["RANK%d" % d for d in windows]
    # fields += [
    #     "($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d)
    #     for d in windows
    # ]
    # names += ["RSV%d" % d for d in windows]
    # fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
    # names += ["IMAX%d" % d for d in windows]
    # fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
    # names += ["IMIN%d" % d for d in windows]
    # fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
    # names += ["IMXD%d" % d for d in windows]
    # fields += ["Corr($close, Log($volume/(Ref($volume, 1)+1e-12)+1), %d)" % d for d in windows]
    # names += ["CORR%d" % d for d in windows]
    # fields += [
    #     "Corr($close/Ref($close,1), Log($volume/(Ref($volume, 1)+1e-12)+1), %d)" % (d,d)
    #     for d in windows
    # ]
    # names += ["CORD%d" % d for d in windows]
    # fields += ["Mean($close>Ref($close, 1), %d)" % (d) for d in windows]
    # names += ["CNTP%d" % d for d in windows]
    # fields += ["Mean($close<Ref($close, 1), %d)" % (d) for d in windows]
    # names += ["CNTN%d" % d for d in windows]
    # fields += [
    #     "(Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d))" % (d,d)
    #     for d in windows
    # ]
    # names += ["CNTD%d" % d for d in windows]
    # fields += [
    #     "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)"
    #     % (d, d)
    #     for d in windows
    # ]
    # names += ["SUMP%d" % d for d in windows]
    # fields += [
    #     "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)"
    #     % (d, d)
    #     for d in windows
    # ]
    # names += ["SUMN%d" % d for d in windows]
    # fields += [
    #     "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
    #     "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
    #     for d in windows
    # ]
    # names += ["SUMD%d" % d for d in windows]
    # fields += ["Mean(Log($volume/(Ref($volume, 1)+1e-12)+1) %d)" % d for d in windows]
    # names += ["VMA%d" % d for d in windows]
    # fields += ["Std(Log($volume/(Ref($volume, 1)+1e-12)+1) %d)" % d for d in windows]
    # names += ["VSTD%d" % d for d in windows]
    # fields += [
    #     "Std((Abs($close/Ref($close, 1)-1)*Log($volume/(Ref($volume, 1)+1e-12)+1))/(Mean(Abs($close/Ref($close, 1)-1)*Log($volume/(Ref($volume, 1)+1e-12)+1), %d)), %d)"
    #     % (d, d)
    #     for d in windows
    # ]
    # names += ["WVMA%d" % d for d in windows]
    # fields += [
    #     "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
    #     % (d, d)
    #     for d in windows
    # ]
    # names += ["VSUMP%d" % d for d in windows]
    # fields += [
    #     "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
    #     % (d, d)
    #     for d in windows
    # ]
    # names += ["VSUMN%d" % d for d in windows]
    # fields += [
    #     "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
    #     "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
    #     for d in windows
    # ]
    # names += ["VSUMD%d" % d for d in windows]
    
    
    # https://www.investopedia.com/terms/r/rateofchange.asp
    # Rate of change, the price change in the past d days, divided by latest close price to remove unit
    fields += ["Ref($close, %d)/$close" % d for d in windows]
    names += ["ROC%d" % d for d in windows]

    # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
    # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
    fields += ["Mean($close, %d)/$close" % d for d in windows]
    names += ["MA%d" % d for d in windows]

    # The standard diviation of close price for the past d days, divided by latest close price to remove unit
    fields += ["Std($close, %d)/$close" % d for d in windows]
    names += ["STD%d" % d for d in windows]

    # The rate of close price change in the past d days, divided by latest close price to remove unit
    # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
    fields += ["Slope($close, %d)/$close" % d for d in windows]
    names += ["BETA%d" % d for d in windows]

    # The R-sqaure value of linear regression for the past d days, represent the trend linear
    fields += ["Rsquare($close, %d)" % d for d in windows]
    names += ["RSQR%d" % d for d in windows]

    # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
    fields += ["Resi($close, %d)/$close" % d for d in windows]
    names += ["RESI%d" % d for d in windows]

    # The max price for past d days, divided by latest close price to remove unit
    fields += ["Max($high, %d)/$close" % d for d in windows]
    names += ["MAX%d" % d for d in windows]

    # The low price for past d days, divided by latest close price to remove unit
    fields += ["Min($low, %d)/$close" % d for d in windows]
    names += ["MIN%d" % d for d in windows]

    # The 80% quantile of past d day's close price, divided by latest close price to remove unit
    # Used with MIN and MAX
    fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
    names += ["QTLU%d" % d for d in windows]

    # The 20% quantile of past d day's close price, divided by latest close price to remove unit
    fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
    names += ["QTLD%d" % d for d in windows]

    # # Get the percentile of current close price in past d day's close price.
    # # Represent the current price level comparing to past N days, add additional information to moving average.
    # fields += ["Rank($close, %d)" % d for d in windows]
    # names += ["RANK%d" % d for d in windows]

    # Represent the price position between upper and lower resistent price for past d days.
    fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
    names += ["RSV%d" % d for d in windows]

    # The number of days between current date and previous highest price date.
    # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
    # The indicator measures the time between highs and the time between lows over a time period.
    # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
    fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
    names += ["IMAX%d" % d for d in windows]

    # The number of days between current date and previous lowest price date.
    # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
    # The indicator measures the time between highs and the time between lows over a time period.
    # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
    fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
    names += ["IMIN%d" % d for d in windows]

    # The time period between previous lowest-price date occur after highest price date.
    # Large value suggest downward momemtum.
    fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
    names += ["IMXD%d" % d for d in windows]

    # The correlation between absolute close price and log scaled trading volume
    fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
    names += ["CORR%d" % d for d in windows]

    # The correlation between price change ratio and volume change ratio
    fields += ["Corr($close/(Ref($close,1)+1e-6), Log($volume/(Ref($volume, 1)+1e-6)+1), %d)" % d for d in windows]
    names += ["CORD%d" % d for d in windows]

    # The percentage of days in past d days that price go up.
    fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
    names += ["CNTP%d" % d for d in windows]

    # The percentage of days in past d days that price go down.
    fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
    names += ["CNTN%d" % d for d in windows]

    # The diff between past up day and past down day
    fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
    names += ["CNTD%d" % d for d in windows]

    # The total gain / the absolute total price changed
    # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
    fields += [
        "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMP%d" % d for d in windows]

    # The total lose / the absolute total price changed
    # Can be derived from SUMP by SUMN = 1 - SUMP
    # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
    fields += [
        "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMN%d" % d for d in windows]

    # The diff ratio between total gain and total lose
    # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
    fields += [
        "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
        "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
        for d in windows
    ]
    names += ["SUMD%d" % d for d in windows]

    # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
    fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
    names += ["VMA%d" % d for d in windows]

    # The standard deviation for volume in past d days.
    fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
    names += ["VSTD%d" % d for d in windows]

    # The volume weighted price change volatility
    fields += [
        "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["WVMA%d" % d for d in windows]

    # The total volume increase / the absolute total volume changed
    fields += [
        "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["VSUMP%d" % d for d in windows]

    # The total volume increase / the absolute total volume changed
    # Can be derived from VSUMP by VSUMN = 1 - VSUMP
    fields += [
        "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["VSUMN%d" % d for d in windows]

    # The diff ratio between total volume increase and total volume decrease
    # RSI indicator for volume
    fields += [
        "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
        "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
        for d in windows
    ]
    names += ["VSUMD%d" % d for d in windows]
    return fields, names
