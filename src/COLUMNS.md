# Which columns did the hotel know on the day of booking?

**Owner:** Data Lead. **Closes:** part of ladder stage 1. **Used by:** `src/features.py`, the PRD and slide 4.

Put every column in exactly one bin, by when the hotel learns its value:

1. **Fixed at booking**: known on the day the booking is made, and it does not change.
2. **Amendable**: known at booking but it can change before arrival, or it is only partly known at booking.
3. **After the fact**: written at check-in, during the stay, or when the outcome is already known.

Only the first bin feeds the MVP. A second-bin column may be tried with its caveat written beside it. A third-bin column never feeds a method. The dataset article (Antonio, de Almeida and Nunes, 2019) says how several columns were recorded; cite it where it settles a bin.

The evidence column holds the number or the source that settled the bin, for example a rate you computed, or a sentence from the dataset article.

| Column | Bin (1, 2 or 3) | Why | Evidence |
|---|---|---|---|
| `hotel` |1|The guest selects the property at the moment of booking.|Standard B2C/B2B booking flow|
| `is_canceled` |3|Target variable. We do not know if they will cancel until they actually cancel or no-show.|Dataset target|
| `lead_time` |1|Calculated chronologically as (arrival date - booking date). Fixed once booked.|""""Antonio et al. (2019) definition""""|
| `arrival_date_year` |1|The guest selects their arrival date at booking.|Standard booking flow|
| `arrival_date_month` |1|The guest selects their arrival date at booking.|Standard booking flow|
| `arrival_date_week_number` |1|Derived directly from the selected arrival date.|Standard booking flow|
| `arrival_date_day_of_month` |1|The guest selects their arrival date at booking.|Standard booking flow|
| `stays_in_weekend_nights` |2|Known initially, but guests can extend or shorten stays prior to arrival.|Standard booking flow|
| `stays_in_week_nights` |2|Known initially, but guests can extend or shorten stays prior to arrival.|Standard booking flow|
| `adults` |2|Party size is stated at booking but can be amended before check-in.|Standard booking flow|
| `children` |2|Party size is stated at booking but can be amended before check-in.|Standard booking flow|
| `babies` |2|Party size is stated at booking but can be amended before check-in.|Standard booking flow|
| `meal` |2|Guests can upgrade meal plans (e.g., add breakfast) prior to arrival.|Standard booking flow|
| `country` |2|Often derived from IP/credit card initially, but officially confirmed/changed via passport at check-in.|Standard booking flow|
| `market_segment` |1|Set by the "door" (e.g., OTA vs Direct) the guest used to make the booking.|B2B/B2C logic in Brief|
| `distribution_channel` |1|Set by how the booking reached the hotel (e.g., Corporate vs Travel Agent).|B2B/B2C logic in Brief|
| `is_repeated_guest` |1|Looked up in the hotel's CRM at the exact moment the booking is created.|Profile history|
| `previous_cancellations` |1|Looked up in the hotel's CRM at the exact moment the booking is created.|Profile history|
| `previous_bookings_not_canceled` |1|Looked up in the hotel's CRM at the exact moment the booking is created.|Profile history|
| `reserved_room_type` |1|The specific room code the guest requested at checkout.|Antonio et al. (2019) definition|
| `assigned_room_type` |3|Handled by front desk operations. Often assigned on the day of arrival.|Antonio et al. (2019) definition|
| `booking_changes` |3|Tally of all amendments made until check-in/cancellation. At booking, this is always 0.|Antonio et al. (2019) definition|
| `deposit_type` |1|Tied directly to the rate/policy selected at the moment of booking.|Brief mentions prepaid rates|
| `agent` |1|The ID of the travel agency making the booking. Fixed.|B2B logic in Brief|
| `company` |1|The ID of the corporate entity making the booking. Fixed.|B2B logic in Brief|
| `days_in_waiting_list` |1|A booking is only placed "on the books" after clearing the waitlist.|Antonio et al. (2019) definition|
| `customer_type` |1|Contract type (e.g., Transient, Group) established at booking.| Standard B2B/B2C logic|
| `adr` |2|Average Daily Rate changes if the guest amends dates, party size, or meals before arrival.|Revenue tree definition|
| `required_car_parking_spaces` |3|While guests can request early, the final count is often only known at physical check-in.|Antonio et al. (2019) definition|
| `total_of_special_requests` |2|Guests can call in to add special requests anytime up until arrival.|Standard booking flow|
| `reservation_status` |3|Target variable leak. The final state (Check-Out, Canceled, No-Show).|Dataset target|
| `reservation_status_date` |3|Target variable leak. The date the final status was set.|Dataset target|

## What happens to a method when the late columns go in?

_Train the same method twice on the same split, once with first-bin columns only and once with the second and third bins added. Then run the first version once more on a random split. Record the three AUCs here and say which one the client should believe._

| Run | Columns | Split | AUC on the proof period |
|---|---|---|---|
| A | First bin only | Time ||
| B | All three bins | Time ||
| C | First bin only | Random ||
