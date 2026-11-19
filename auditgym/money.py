from decimal import ROUND_HALF_EVEN, Decimal, localcontext

EXPONENTS = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "KZT": 2,
    "CHF": 2,
    "AED": 2,
    "TRY": 2,
    "JPY": 0,
    "KWD": 3,
}

USD_VALUE = {
    "USD": "1",
    "EUR": "1.08",
    "GBP": "1.27",
    "KZT": "0.00205",
    "CHF": "1.12",
    "AED": "0.2723",
    "TRY": "0.029",
    "JPY": "0.0067",
    "KWD": "3.25",
}


def convert(amount_minor, source, target, rates):
    if source == target:
        return amount_minor
    with localcontext() as ctx:
        ctx.prec = 50
        major = Decimal(amount_minor).scaleb(-EXPONENTS[source])
        value = major * rates[source] / rates[target]
        return int(value.scaleb(EXPONENTS[target]).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def rate_text(value):
    return format(Decimal(f"{value:.6g}"), "f")


def fx_id(day, currency, base):
    return f"{day}:{currency}:{base}"
