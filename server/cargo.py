import asyncio
from datetime import date, timedelta
from flight_intel_patch import validate_extraction_results

async def test_cargo():
    # Use "today" and "tomorrow" so this stays fresh whenever you run it
    today = date.today()
    tomorrow = today + timedelta(days=1)

    extraction = {
        "flights": [
            {
                # UPS 5X488: Louisville (SDF) -> Toronto (YYZ)
                "date": today.strftime("%m/%d/%Y"),
                "flight_no": "5X488",        # UPS cargo
                "origin": "SDF",             # UPS hub
                "dest": None,                # Let the API fill this in
                "sched_out_local": None,
                "sched_in_local": None,
            },
            {
                # Delta DL3913: Seattle (SEA) -> Portland (PDX)
                "date": tomorrow.strftime("%m/%d/%Y"),
                "flight_no": "DL3913",       # Delta passenger
                "origin": "SEA",
                "dest": None,                # Let the API fill this in
                "sched_out_local": None,
                "sched_in_local": None,
            },
        ]
    }

    result = await validate_extraction_results(extraction)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for flight in result["enriched_flights"]:
        print(f"\n✈️  {flight['flight_no']} on {flight['date']}")
        print(f"   Route: {flight.get('origin', 'N/A')} → {flight.get('dest', 'N/A')}")
        print(f"   Times: {flight.get('sched_out_local', 'N/A')} → {flight.get('sched_in_local', 'N/A')}")
        if flight.get("validation_result"):
            vr = flight["validation_result"]
            print(f"   Source: {vr['source']} | Confidence: {vr['confidence']:.2f}")
            print(f"   Filled: {list(vr['filled_fields'].keys())}")

if __name__ == "__main__":
    asyncio.run(test_cargo())
